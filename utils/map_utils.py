import folium
from streamlit_folium import st_folium

def render_map(lat, lon, hazards=[]):
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid lat/lon values for map: {lat}, {lon}")

    fmap = folium.Map(location=[lat, lon], zoom_start=16)

    # user’s current GPS position
    folium.Marker(
        [lat, lon],
        tooltip="Current Position",
        icon=folium.Icon(color='blue', icon='car', prefix='fa')
    ).add_to(fmap)

    # pothole hazard markers
    for h in hazards:
        try:
            h_lat = float(h.get('lat'))
            h_lon = float(h.get('lon'))
            folium.Marker(
                [h_lat, h_lon],
                popup=f"Pothole Detected (ID: {h.get('id', 'N/A')})",
                icon=folium.Icon(color='red', icon='warning', prefix='fa')
            ).add_to(fmap)
        except (TypeError, ValueError):
            print(f"⚠️ Skipping invalid hazard location: {h}")

    return fmap
