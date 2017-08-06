#! /usr/bin/env python

import os
import argparse
import numpy as np
from scipy.spatial import Delaunay
from math import sqrt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f')
parser.add_argument('--output', '-o')
args = parser.parse_args()

img = cv2.imread(args.filename)

def display(name, img):
  if args.output:
    path = args.output + '/' + name + '.png'
    try:
      os.makedirs(path)
    except:
      pass
    cv2.imwrite(path, img)
  cv2.imshow(name, img)

def averageColor(a, b, c):
  contour = np.array([[a], [b], [c]])
  count = 0
  color = np.array([0, 0, 0])
  maxWidth = max(a[0], b[0], c[0])
  minWidth = min(a[0], b[0], c[0])
  maxHeight = max(a[1], b[1], c[1])
  minHeight = min(a[1], b[1], c[1])
  for x in range(minWidth, maxWidth):
    for y in range(minHeight, maxHeight):
      if cv2.pointPolygonTest(contour, (x, y), False) > -1:
        count += 1
        color += img[y, x]
  return color / count

def draw(_):
  blur = cv2.getTrackbarPos('blur', 'blur')
  blurred = cv2.blur(img, (blur, blur))
  display('blur', blurred)
  Z = blurred.reshape((-1,3))

  # convert to np.float32
  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = cv2.getTrackbarPos('cluster', 'cluster')
  ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  clustered = center[label.flatten()].reshape(img.shape)
  display('cluster', clustered)

  threshold1 = cv2.getTrackbarPos('threshold1', 'canny')
  threshold2 = cv2.getTrackbarPos('threshold2', 'canny')
  cannied = cv2.Canny(clustered, threshold1, threshold1)
  display('canny', cannied)

  contours, _ = cv2.findContours(cannied, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  epsilon = cv2.getTrackbarPos('epsilon', 'contour')
  area = cv2.getTrackbarPos('area', 'contour')
  contours = [cv2.approxPolyDP(contour, epsilon, False) for contour in contours if cv2.contourArea(contour) > area]
  contoured = cannied.copy()
  contoured[:,:] = 0
  cv2.drawContours(contoured, contours, -1, 255, 1)
  display('contour', contoured)

  geometry = contoured.copy()
  geometry[:,:] = 0
  height, width = geometry.shape
  points = [np.array([0, 0]), np.array([0, height]), np.array([width, 0]), np.array([width, height])]
  maxDistance = cv2.getTrackbarPos('distance', 'geometry')
  for contour in contours:
    for point in contour:
      for other in points:
        distance = sqrt((other[0] - point[0][0])**2 + (other[1] - point[0][1])**2)
        if distance < maxDistance:
          break
      else:
        points.append(point[0])
  triangles = np.array(points)[Delaunay(points).simplices]
  for [a, b, c] in triangles:
    cv2.line(geometry, tuple(a), tuple(b), 125)
    cv2.line(geometry, tuple(b), tuple(c), 125)
    cv2.line(geometry, tuple(c), tuple(a), 125)
  display('geometry', geometry)

  final = img.copy()
  print 'Finding colors', len(triangles)
  for i, triangle in enumerate(triangles):
    print i
    color = averageColor(*triangle)
    cv2.fillPoly(final, [triangle], color)
  display('final', final)

cv2.namedWindow('blur')
cv2.namedWindow('cluster')
cv2.namedWindow('canny')
cv2.namedWindow('contour')
cv2.namedWindow('geometry')
cv2.namedWindow('final')
cv2.createTrackbar('blur', 'blur', 10, 10, draw)
cv2.createTrackbar('cluster', 'cluster', 3, 32, draw)
cv2.createTrackbar('threshold1', 'canny', 100, 500, draw)
cv2.createTrackbar('threshold2', 'canny', 300, 500, draw)
cv2.createTrackbar('epsilon', 'contour', 5, 50, draw)
cv2.createTrackbar('area', 'contour', 0, 500, draw)
cv2.createTrackbar('distance', 'geometry', 5, 50, draw)

draw(None)

cv2.waitKey(0)
cv2.destroyAllWindows()
