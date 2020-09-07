from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np


def index(request):
    return render(request, 'index.html')


def readImage(request):
    request.encoding = 'utf-8'
    f = request.FILES.get('srcImage', None)
    content = f.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./photo.jpg', img)
    return HttpResponse('teddddddddddd')
