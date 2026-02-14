from django.urls import path

from .views import predict_api, home, history_view

urlpatterns = [
    path('', home, name='home'),
    path('api/predict/', predict_api),
    path('history/', history_view, name='history'),
]
