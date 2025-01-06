
def convert_pixels_to_meters(pixels_dist, reference_height_in_meters, reference_height_in_pixels):
    return pixels_dist * reference_height_in_meters / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    return meters * reference_height_in_pixels / reference_height_in_meters