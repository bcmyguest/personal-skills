"""Session auth helpers."""
from functools import wraps

from django.http import JsonResponse


def current_user(request):
    """Return the authenticated user dict, or None."""
    return getattr(request, "user", None)


def login_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if current_user(request) is None:
            return JsonResponse({"error": "authentication required"}, status=401)
        return view(request, *args, **kwargs)

    return wrapper
