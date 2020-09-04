from django.shortcuts import render
from django.http import HttpResponse
import requests
from bs4 import BeautifulSoup
import re


def index(request):
    return HttpResponse('<form action="answer" method="get"><p>name: <input type="text" '
                        'name="moviename" /></p><input type="submit" value="Submit" /></form>')


def answer(request):
    if request.method == 'GET':
        moviename = request.GET.get('moviename', 'aaa')
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/75.0.3770.142 Safari/537.36'}
        url = 'https://www.douban.com/search?q={}'.format(moviename)
        index_response = requests.get(url=url, headers=headers)
        bf = BeautifulSoup(index_response.text, 'html5lib')
        films = bf.find_all('div', class_='result')
        link = films[0].find_all('a')[0]
        ans = re.findall(r'"(.+?)"', str(link))[1]
        index_response = requests.get(url=ans, headers=headers)
        bf = BeautifulSoup(index_response.text, 'html5lib')
        alsolike = bf.find_all('div', class_='recommendations-bd')
        assert len(alsolike) == 1
        films = re.findall(r'alt="(.+?)"', str(alsolike))
        films = ['<td>'+film+'</td>' for film in films]
        return HttpResponse('<table border="1"><tr>' + ' '.join(films) + '</tr></table>')
