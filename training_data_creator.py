from osgeo import osr, gdal
import numpy as np
import scandir
import random
import math
import cv2
import re


image = "Large.tif"
filename = "craters.txt"

minSize = 100

negSize = 50
count = 500

__transform = None
__transform_inv = None

__wgs84_wkt = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""



# Function to convert a lat, long position to pixel coordinates on an image
def __lat_lon_to_pixels(lat, lon, sc, off):
    global __transform
    (x, y, e) = __transform.TransformPoint(lon, lat)

    x = max(0, int((x * sc[0]) + off[0]))
    y = max(0, int((y * sc[1]) + off[1]))
    return (x,y)

# Function to determine the bounding box of a camera FOV at a given altitude
def __m_to_gps(m, lat):
    # m to degrees long/lat
    xScale = 1.0/(111111*math.cos(lat))
    yScale = 1.0/111111

    D = math.tan(self.__fov/2)*2*alt

    x = math.sqrt( (D*D) / (1 + (self.__ratio*self.__ratio)) )
    y = x*self.__ratio
    # x and y are currently in meters

    x = x * xScale
    y = y * yScale
    # x and y are now in degrees long/lat

    return (x, y)


# Open the image to get the geo data
src = gdal.Open(image)
# Load the image with opencv
img = cv2.imread(image)



# The source projection (wgs84, coordinate system used by gps)
source = osr.SpatialReference()
source.ImportFromWkt(__wgs84_wkt)

# The target projection (whatever is used by the geotiff image)
target = osr.SpatialReference()
target.ImportFromWkt(src.GetProjection())

# Create the transform and an inverse transform
__transform = osr.CoordinateTransformation(source, target)
__transform_inv = osr.CoordinateTransformation(target, source)



# Get the coordinates of the upper left and lower right point of the image
# (uses whatever coordinate system the image was encoded with)
ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
lrx = ulx + (src.RasterXSize * xres)
lry = uly + (src.RasterYSize * yres)

# Get the height and width of the image (in pixels)
(hei, wid, cha) = img.shape

# Determine values to interpolate between the geo coordinate system of
# the image and pixel values
xScale = wid/(lrx-ulx)
yScale = hei/(lry-uly)

xOff = -ulx*xScale
yOff = -uly*yScale

# Use the transform generated earlier to convert the coordinate system of the
# image to GPS (wgs84) to determine the GPS boundaries of the image
(minlon, maxlat, e) = __transform_inv.TransformPoint(ulx, uly)
(maxlon, minlat, e) = __transform_inv.TransformPoint(lrx, lry)








coords = []

input_file = open(filename)


rows = re.split(r'\r', input_file.readline())
for row in rows[1:]:
    cols = re.split(r'\t', row)
    coords.append( (float(cols[1]), float(cols[2]), float(cols[5])) )

input_file.close()






i = 0
r = 20

for coord in coords:
    (x,y) = __lat_lon_to_pixels(coord[0], coord[1], (xScale, yScale), (xOff, yOff))
    if x > 1 and x < img.shape[1] and y > 1 and y < img.shape[0]:
        r = int(coord[2]*4)
        #cv2.circle(img, (x,y), 2*r, (0,255,0), -1)
        cropped = img[y-r:y+r, x-r:x+r, :]
        (h,w,c) = cropped.shape
        if w > minSize and h > minSize:
            cv2.imwrite("training_data/positive/crater_"+str(i)+".png", cropped)
            i = i + 1
            if i > count:
                break

(h,w,c) = img.shape
for i in range(count):
    x = random.randint(negSize, w-negSize)
    y = random.randint(negSize, h-negSize)
    cropped = img[y-negSize:y+negSize, x-negSize:x+negSize, :]
    cv2.imwrite("training_data/negative/not_crater"+str(i)+".png", cropped)

exit()



img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)

cv2.imshow("image", img)
cv2.waitKey(0)

# Each image is saved in the list along with the following information
#imgList.append( [img, (minlat, maxlat), (minlon, maxlon), (xScale, yScale), (xOff, yOff)] )
