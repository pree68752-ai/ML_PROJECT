from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.core.files.storage import default_storage
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import timedelta

from .ml_engine import predict_image
from .models import Prediction

import os


# ---------------- HOME ----------------

@login_required
def home(request):
    return render(request, "home.html")


# ---------------- PREDICT API ----------------

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_api(request):

    image = request.FILES.get('image')

    if not image:
        return Response({"error": "No image uploaded"}, status=400)

    # Save temporary file
    path = default_storage.save("temp.jpg", image)
    full_path = os.path.join("media", path)

    # Run ML prediction
    result = predict_image(full_path)

    # 🔥 Auto delete predictions older than 30 days
    expiry_date = timezone.now() - timedelta(days=30)
    Prediction.objects.filter(created_at__lt=expiry_date).delete()

    # Save new prediction
    Prediction.objects.create(
        user=request.user,
        image_name=image.name,
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

    # Delete temp file
    default_storage.delete(path)

    return Response(result)


# ---------------- SIGNUP ----------------

def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()

    return render(request, "signup.html", {"form": form})


# ---------------- HISTORY ----------------

@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, "history.html", {"predictions": predictions})

@login_required
def profile_view(request):
    return render(request, "profile.html")
