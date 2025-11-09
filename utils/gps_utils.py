import geocoder

def get_current_gps():
    try:
        g = geocoder.ip('me')
        if g.latlng and isinstance(g.latlng, list):
            lat, lon = g.latlng
            return {'latitude': float(lat), 'longitude': float(lon)}
    except Exception as e:
        print("GPS fetch error:", e)

    # Return None only if GPS fails
    return {'latitude': None, 'longitude': None}
