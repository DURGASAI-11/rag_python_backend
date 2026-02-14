from jose import jwt, JWTError
from fastapi import HTTPException, Request

ALGORITHM = "RS256"

# Load public key once at startup
with open("app/keys/access-token.public.key", "r") as f:
    ACCESS_PUBLIC_KEY = f.read()


async def verify_access_token(request: Request):
    token = None

    #  Check Authorization header first 
    auth_header = request.headers.get("Authorization")

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]

    # Fallback to cookie 
    elif "accessToken" in request.cookies:
        token = request.cookies.get("accessToken")

    if not token:
        raise HTTPException(status_code=401, detail="Access token missing")

    try:
        payload = jwt.decode(
            token,
            ACCESS_PUBLIC_KEY,
            algorithms=[ALGORITHM]
        )

        # Optional: Validate required claims
        if "userId" not in payload:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        return {
            "user_id": payload["userId"],
            "role": payload.get("role")
        }

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
