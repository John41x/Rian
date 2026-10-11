# Spec 10: Multi-Tenant JWT Authentication & Role-Based Access Control (RBAC)

## 1. Module Overview & Purpose
Secures the Rian platform with JSON Web Token (JWT) authentication and Role-Based Access Control (RBAC). 
Operators, field patrol responders, and system administrators receive scoped credentials to prevent unauthorized tampering with perimeter sensors, intercept directives, or camera telemetry.

### Roles & Permissions:
1. `ADMIN`: Full system access: user management, camera provisioning, SLA benchmarking, and system settings.
2. `OPERATOR`: Common Operating Picture (COP) access, acknowledge/resolve directives, query Tactical Copilot, run simulations.
3. `RESPONDER`: Field access: view active assigned intercept directives and navigation routes; update on-scene status.

---

## 2. Dependencies & Reference Architecture
- **Reference Repo:** `fastapi/full-stack-fastapi-template` (`app/core/security.py`, OAuth2 password bearer).
- **Libraries:** `pyjwt` or `python-jose`, `passlib[bcrypt]` or `hashlib` (PBKDF2/Argon2/bcrypt).
- **Target Module Path:** `src/common/auth.py`, `src/web/auth_routes.py`.
- **Test Path:** `tests/test_common/test_auth.py`, `tests/test_web/test_auth_api.py`.

---

## 3. Data Contracts & Database Schema

### 3.1 SQL Schema (`users` table)
```sql
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(128),
    role VARCHAR(32) NOT NULL DEFAULT 'OPERATOR', -- 'ADMIN', 'OPERATOR', 'RESPONDER'
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
```

### 3.2 Default Bootstrapped Admin
On system initialization, if no admin account exists, Rian creates a default initial administrator:
- Email: `admin@rian.internal` (or `settings.FIRST_SUPERUSER`)
- Password: `rian_admin_secret` (or `settings.FIRST_SUPERUSER_PASSWORD`)
- Role: `ADMIN`

---

## 4. API Endpoints

- `POST /api/v1/auth/login`: Accepts OAuth2 password form (`username` (email), `password`). Returns `{ "access_token": "...", "token_type": "bearer", "user": { "email": "...", "role": "..." } }`.
- `GET /api/v1/auth/me`: Returns the authenticated user record from the Bearer token.
- `POST /api/v1/admin/users`: Creates a new user account (Admin only).
- `GET /api/v1/admin/users`: Lists all users (Admin only).
- `DELETE /api/v1/admin/users/{user_id}`: Deactivates or removes a user (Admin only).

---

## 5. Acceptance Criteria
- [ ] Passwords hashed with salted SHA256/PBKDF2 or bcrypt; plain text passwords never stored.
- [ ] JWT tokens expire after configurable duration (default 24 hours).
- [ ] `ADMIN`, `OPERATOR`, and `RESPONDER` roles enforced on protected endpoints via FastAPI dependencies.
- [ ] 100% test pass rate in `tests/test_common/test_auth.py`.
