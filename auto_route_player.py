#!/usr/bin/env python3
import sys
import time
import random
import uuid
import requests
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CoordSource(Enum):
    PLANNED = "planned"
    STEPS = "steps"
    AREA = "area"


@dataclass
class Config:
    CHIRP_BASE: str = "https://chirp.kantegory.me/api/geo"
    DM_BASE: str = "https://chirp.kantegory.me/api/device-management"
    TRACCAR_OSMAND: str = "http://89.104.68.209:5055/"
    DEFAULT_SPEED_KMH: float = 40.0
    DEFAULT_INTERVAL_S: float = 0.5
    DEFAULT_STEP_M: float = 150.0
    DEFAULT_DEVIATE_M: float = 10.0
    MOSCOW_BBOX = {
        "lat_min": 55.60, "lat_max": 55.90,
        "lon_min": 37.40, "lon_max": 37.80
    }
    EARTH_RADIUS_M: float = 6371000.0
    METERS_PER_DEGREE_LAT: float = 111320.0


class ChirpAPI:
    def __init__(self, token: str, config: Config = None):
        self.token = token
        self.config = config or Config()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        })
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """универсальный метод для запросов с обработкой ошибок"""
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ошибка запроса к {url}: {e}")
    
    def get_routes(self) -> List[Dict[str, Any]]:
        """получить список маршрутов"""
        response = self._make_request("GET", f"{self.config.CHIRP_BASE}/routes")
        routes = response.json()
        return routes[:5] if isinstance(routes, list) else []
    
    def get_route(self, route_id: int) -> Dict[str, Any]:
        """получить конкретный маршрут"""
        response = self._make_request("GET", f"{self.config.CHIRP_BASE}/routes/{route_id}")
        return response.json()
    
    def get_device(self, device_id: int) -> Dict[str, Any]:
        """получить информацию об устройстве"""
        response = self._make_request("GET", f"{self.config.DM_BASE}/devices/{device_id}")
        return response.json()
    
    def get_devices_names(self, with_routes: bool = False) -> List[Dict[str, Any]]:
        """получить список устройств"""
        params = {"with_routes": str(with_routes).lower()}
        response = self._make_request("GET", f"{self.config.DM_BASE}/devices/names", params=params)
        return response.json() or []
    
    def create_route(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        """создать новый маршрут"""
        self.session.headers.update({"Content-Type": "application/json"})
        response = self._make_request("POST", f"{self.config.CHIRP_BASE}/routes", json=route_data)
        return response.json()




def get_route_first_device_unique_id(api: ChirpAPI, route_id: int) -> Optional[str]:
    """получить uniqueId первого устройства маршрута"""
    try:
        route_data = api.get_route(route_id)
        devices = route_data.get("devices") or []
        if not devices:
            return None
        dev_id = devices[0].get("id")
        if not dev_id:
            return None
        device_info = api.get_device(dev_id)
        return device_info.get("uniqueId")
    except RuntimeError:
        return None


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float, earth_radius: float = Config.EARTH_RADIUS_M) -> float:
    """расстояние между двумя точками в метрах по формуле гаверсинуса"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return earth_radius * c


def flatten_with_step(coords: List[List[float]], step_m: float) -> List[Tuple[float, float]]:
    """дискретизация маршрута с заданным шагом в метрах"""
    if not coords:
        return []
    
    pts = [(float(coord[0]), float(coord[1])) for coord in coords]
    if len(pts) <= 1:
        return pts
    
    result = [pts[0]]
    for i in range(len(pts)-1):
        lat1, lon1 = pts[i][1], pts[i][0]
        lat2, lon2 = pts[i+1][1], pts[i+1][0]
        dist = haversine_distance_m(lat1, lon1, lat2, lon2)
        steps = max(1, int(dist//step_m))
        for k in range(1, steps+1):
            t = k/steps
            lat = lat1 + (lat2-lat1)*t
            lon = lon1 + (lon2-lon1)*t
            result.append((lon, lat))
    return result


def extract_planned_or_steps(route_json: Dict[str, Any], source: CoordSource) -> Optional[List[List[float]]]:
    """извлечь координаты из маршрута по указанному источнику"""
    if source == CoordSource.PLANNED:
        coords = (((route_json.get("planned_route") or {}).get("geometry") or {}).get("coordinates"))
        if coords:
            return coords
    elif source == CoordSource.STEPS:
        legs = ((route_json.get("planned_route") or {}).get("legs") or [])
        coords: List[List[float]] = []
        for leg in legs:
            for st in (leg.get("steps") or []):
                gc = (((st.get("geometry") or {}).get("coordinates")) or [])
                coords.extend(gc)
        return coords if coords else None
    elif source == CoordSource.AREA:
        return (((route_json.get("area") or {}).get("geometry") or {}).get("coordinates"))
    return None


def send_osmand(unique_id: str, lat: float, lon: float, timestamp_sec: Optional[int] = None, 
                speed_kmh: float = None, config: Config = None) -> int:
    """отправить точку в traccar"""
    config = config or Config()
    speed_kmh = speed_kmh or config.DEFAULT_SPEED_KMH
    params = {"id": unique_id, "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "speed": f"{speed_kmh:.2f}"}
    # если timestamp не указан — пусть traccar проставит серверное время
    if timestamp_sec is not None:
        params["timestamp"] = str(timestamp_sec)
    r = requests.get(config.TRACCAR_OSMAND, params=params, timeout=30)
    return r.status_code


def arrive_to_point_b(unique_id: str, coords: List[Tuple[float, float]], config: Config = None) -> Tuple[float, float, int]:
    """отправляет только финальную точку B с нулевой скоростью; возвращает (lat, lon, http_code)"""
    if not coords:
        raise RuntimeError("нет координат для точки B")
    last_lon, last_lat = coords[-1]
    code = send_osmand(unique_id, last_lat, last_lon, None, 0.0, config)
    return last_lat, last_lon, code


def offset_point(lat: float, lon: float, delta_m: float, bearing_deg: float = 45.0, 
                 config: Config = None) -> Tuple[float, float]:
    """смещает точку на delta_m метров под углом bearing_deg и возвращает (lat, lon)"""
    config = config or Config()
    if delta_m <= 0:
        return lat, lon
    R_lat = config.METERS_PER_DEGREE_LAT
    R_lon = config.METERS_PER_DEGREE_LAT * max(0.000001, math.cos(math.radians(lat)))
    rad = math.radians(bearing_deg)
    dx = delta_m * math.cos(rad)
    dy = delta_m * math.sin(rad)
    dlat = dy / R_lat
    dlon = dx / R_lon
    return lat + dlat, lon + dlon


def interpolate_between(lon1: float, lat1: float, lon2: float, lat2: float, t: float) -> Tuple[float, float]:
    """линейная интерполяция между двумя точками [lon, lat] при t∈[0..1]"""
    t = max(0.0, min(1.0, t))
    lon = lon1 + (lon2 - lon1) * t
    lat = lat1 + (lat2 - lat1) * t
    return lon, lat


def find_unassigned_device_id(api: ChirpAPI) -> Optional[int]:
    """находит id первого устройства без привязанных маршрутов"""
    try:
        devices = api.get_devices_names(with_routes=False)
        # пропускаем устройства с "UpdatedDevice" в имени
        for device in devices:
            if not isinstance(device, dict):
                continue
            name = str(device.get("name") or "")
            if "UpdatedDevice" in name:
                continue
            dev_id = device.get("id")
            if dev_id is not None:
                try:
                    return int(dev_id)
                except (ValueError, TypeError):
                    continue
    except RuntimeError:
        pass
    return None


def create_small_moscow_route(api: ChirpAPI, device_id: int, name: str = "test route", 
                             description: str = "test route") -> Optional[int]:
    """создаёт небольшой случайный маршрут в москве"""
    # генерируем случайные координаты в москве
    base_lat = 55.7558 + random.uniform(-0.01, 0.01)
    base_lon = 37.6176 + random.uniform(-0.01, 0.01)
    coords = [[base_lon, base_lat]]
    
    bearing = random.uniform(0, 360)
    for _ in range(random.randint(2, 5)):
        bearing += random.uniform(-45, 45)
        dist = random.choice([120.0, 160.0, 200.0]) * random.uniform(0.6, 1.4)
        lat_prev, lon_prev = coords[-1][1], coords[-1][0]
        lat_next, lon_next = offset_point(lat_prev, lon_prev, dist, bearing, api.config)
        # ограничиваем bbox москвы
        lat_next = max(api.config.MOSCOW_BBOX["lat_min"], min(api.config.MOSCOW_BBOX["lat_max"], lat_next))
        lon_next = max(api.config.MOSCOW_BBOX["lon_min"], min(api.config.MOSCOW_BBOX["lon_max"], lon_next))
        coords.append([lon_next, lat_next])
    
    area_id = f"auto_small_route_moscow_{uuid.uuid4().hex[:8]}"
    payload: Dict[str, Any] = {
        "name": f"{name} {time.strftime('%H:%M:%S')}",
        "description": description,
        "attributes": {},
        "device_id": int(device_id),
        "area": {
            "id": area_id,
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": coords},
        },
    }
    try:
        result = api.create_route(payload)
        return int(result.get("id")) if result.get("id") else None
    except RuntimeError as e:
        print(f"не удалось создать маршрут: {e}")
        return None


def get_token() -> str:
    """получить токен от пользователя"""
    try:
        token = input("bearer токен chirp.kantegory.me: ").strip()
        if not token:
            raise ValueError("нужен токен")
        return token
    except KeyboardInterrupt:
        print()
        sys.exit(130)


def get_default_params() -> Dict[str, Any]:
    """параметры проигрывания по умолчанию"""
    return {
        "route_id": 0,
        "desired_points": None,  # все точки
        "interval_s": 0.5,  # 0.5 сек
        "speed_kmh": 40.0,  # 40 км/ч
        "step_m": None,  # без дискретизации
        "only_b": False,  # полный маршрут
        "skip_first_10": False,  # не пропускать
        "deviate": False,  # без отклонений
        "deviate_m": 10.0,  # 10м если нужно
        "start_between": False,  # начинать с точки A
        "start_between_t": 0.5,  # 50% если нужно
        "source": CoordSource.PLANNED
    }


def show_available_routes(api: ChirpAPI):
    """показать доступные маршруты"""
    routes = api.get_routes()
    if routes:
        print("\nдоступные маршруты:")
        print("id\tstatus\tdist(km)\tdevices\tname")
        for route in routes[:5]:
            rid = route.get("id")
            name = (route.get("name") or "").strip()
            status = route.get("status") or ""
            dist_km = round(float(route.get("distance") or 0.0) / 1000.0, 1)
            devices = route.get("devices") or []
            dev_id = devices[0].get("id") if devices else None
            uniq_flag = "-"
            if dev_id:
                try:
                    device_info = api.get_device(dev_id)
                    if device_info.get("uniqueId"):
                        uniq_flag = "uid"
                except RuntimeError:
                    pass
            print(f"{rid}\t{status}\t{dist_km}\t{dev_id or '-'}:{uniq_flag}\t{name}")
    else:
        print("(маршрутов нет)")


def create_test_route(api: ChirpAPI) -> Optional[int]:
    """создать тестовый маршрут в москве"""
    # пробуем найти свободный device_id автоматически
    auto_dev_id = find_unassigned_device_id(api)
    if auto_dev_id is None:
        dev_id_str = input("device_id для маршрута (свободный): ").strip()
        auto_dev_id = int(dev_id_str)
    
    name = (input("имя маршрута (по умолчанию 'test route'): ").strip() or "test route")
    desc = (input("описание (по умолчанию 'test route'): ").strip() or "test route")
    
    try:
        created_route_id = create_small_moscow_route(api, int(auto_dev_id), name, desc)
        if created_route_id:
            print(f"создан маршрут id={created_route_id} (device_id={auto_dev_id})")
        return created_route_id
    except Exception as ce:
        print(f"ошибка создания маршрута: {ce}")
        return None




def process_route_coordinates(route_json: Dict[str, Any], params: Dict[str, Any], config: Config) -> List[Tuple[float, float]]:
    """обработать координаты маршрута согласно параметрам"""
    # извлекаем координаты
    coords = extract_planned_or_steps(route_json, params["source"])
    if not coords:
        coords = extract_planned_or_steps(route_json, CoordSource.AREA)
    if not coords:
        raise RuntimeError("не удалось извлечь координаты из маршрута")
    
    # дискретизация по дистанции если нужно
    if params["step_m"]:
        discretized = flatten_with_step(coords, params["step_m"])
        base_pts = [(float(lon), float(lat)) for (lon, lat) in discretized]
    else:
        base_pts = [(float(lon), float(lat)) for lon, lat in coords]
    
    # выборка точек
    if params["desired_points"] is None or params["desired_points"] >= len(base_pts):
        sampled = base_pts
    else:
        idxs = [round(i * (len(base_pts) - 1) / (params["desired_points"] - 1)) for i in range(params["desired_points"])]
        sampled = [base_pts[i] for i in idxs]
    
    # отклонение с маршрута
    if not params["only_b"] and params["deviate"] and params["deviate_m"] > 0.0:
        steps_coords = extract_planned_or_steps(route_json, CoordSource.STEPS)
        if steps_coords:
            densified = flatten_with_step(steps_coords, max(1.0, params["deviate_m"]))
            if densified:
                sampled = [(float(lon), float(lat)) for (lon, lat) in densified]
    
    # старт между точками
    if not params["only_b"] and params["start_between"] and len(sampled) >= 2:
        a_lon, a_lat = sampled[0]
        n_lon, n_lat = sampled[1]
        new_lon, new_lat = interpolate_between(a_lon, a_lat, n_lon, n_lat, params["start_between_t"])
        sampled = [(new_lon, new_lat)] + sampled[1:]
        # убираем дубликаты исходной точки A
        def _not_a(pt: Tuple[float, float]) -> bool:
            return abs(pt[0] - a_lon) > 1e-9 or abs(pt[1] - a_lat) > 1e-9
        sampled = [pt for pt in sampled if _not_a(pt)]
    
    # пропуск первых точек
    if not params["only_b"] and params["skip_first_10"] and len(sampled) > 10:
        sampled = sampled[10:]
    
    return sampled


def execute_route_playback(unique_id: str, sampled: List[Tuple[float, float]], params: Dict[str, Any], config: Config):
    """выполнить проигрывание маршрута"""
    print(f"uniqueId={unique_id}, sending {len(sampled)} точек")
    
    if params["only_b"]:
        # только финальная точка
        b_lat, b_lon, code = arrive_to_point_b(unique_id, sampled, config)
        print(f"#only B {b_lat:.6f},{b_lon:.6f} (stop) -> {code}")
    else:
        # полный маршрут
        for i, (lon, lat) in enumerate(sampled):
            code = send_osmand(unique_id, lat, lon, None, params["speed_kmh"], config)
            print(f"#{i+1}/{len(sampled)} {lat:.6f},{lon:.6f} -> {code}")
            time.sleep(params["interval_s"])
        # финальная точка с нулевой скоростью
        if sampled:
            b_lat, b_lon, code = arrive_to_point_b(unique_id, sampled, config)
            print(f"#final B {b_lat:.6f},{b_lon:.6f} (stop) -> {code}")
    print("done")


def main():
    try:
        # получить токен
        token = get_token()
        
        # инициализация api
        config = Config()
        api = ChirpAPI(token, config)
        
        # спросить про создание тестового маршрута
        try_create_str = input("создать небольшой маршрут в Москве и использовать его? [y/N]: ").strip().lower()
        if try_create_str in ("y", "yes", "д", "да"):
            created_route_id = create_test_route(api)
            if created_route_id:
                route_id = created_route_id
            else:
                route_id = int(input("\nвведи route id, который проигрывать: ").strip())
        else:
            # показать доступные маршруты
            show_available_routes(api)
            route_id = int(input("\nвведи route id, который проигрывать: ").strip())
        
        # параметры по умолчанию
        params = get_default_params()
        params["route_id"] = route_id
        
        # получение данных маршрута
        route_json = api.get_route(route_id)
        unique_id = get_route_first_device_unique_id(api, route_id)
        if not unique_id:
            raise RuntimeError("у маршрута нет устройства с uniqueId")
        
        # обработка координат
        sampled = process_route_coordinates(route_json, params, config)
        
        # выполнение проигрывания
        execute_route_playback(unique_id, sampled, params, config)
        
    except Exception as e:
        print(f"ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


