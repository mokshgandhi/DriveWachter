import folium
from streamlit_folium import st_folium

def render_map(lat, lon, hazards=[]):
    # Create folium map centered at current location
    fmap = folium.Map(location=[lat, lon], zoom_start=16)
    
    # Add user’s current GPS position
    folium.Marker(
        [lat, lon],
        tooltip="Current Position",
        icon=folium.Icon(color='blue', icon='car', prefix='fa')
    ).add_to(fmap)
    
    # Add pothole hazard markers
    for h in hazards:
        folium.Marker(
            [h['lat'], h['lon']],
            popup=f"Pothole Detected (ID: {h['id']})",
            icon=folium.Icon(color='red', icon='warning', prefix='fa')
        ).add_to(fmap)

    return fmap
