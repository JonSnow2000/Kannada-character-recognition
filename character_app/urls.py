from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_character, name='predict_character'),
]
