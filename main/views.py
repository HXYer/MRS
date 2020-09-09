from django.shortcuts import render
import cv2
import numpy as np
from .evaluate import predict


def index(request):
    return render(request, 'index.html')


def readImage(request):
    request.encoding = 'utf-8'
    f = request.FILES.get('srcImage', None)
    content = f.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./main/static/test/eval.jpg', img)
    ans = predict('./main/static/test/eval.jpg')
    ans = str(ans[0]) + str(ans[1]) + '.' + str(ans[2]) + str(ans[3])
    return render(request, 'index.html', {'havephoto': True, 'ans': ans})
