import geocoder

def get_current_gps():
    g = geocoder.ip('me')
    if g.latlng:
        return {'latitude': g.latlng[0], 'longitude': g.latlng[1]}
    else:
        return {'latitude': None, 'longitude': None}
