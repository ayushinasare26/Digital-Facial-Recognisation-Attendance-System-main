# Geo-Verified Facial Recognition Attendance System for Enterprise Workforce Management

An enterprise-grade, full-stack biometric attendance and workforce governance platform engineered for organizations with multiple physical work sites, shift-based staff, and remote/field personnel. Built with high-precision face recognition (128-dimensional deep metric embeddings), interactive multi-frame liveness verification, Haversine geofence validation, industrial shift rules, role-scoped portals, and payroll-ready reporting.

---

## 🏢 Architectural Overview: Three-Portal Division

The system enforces strict multi-tenant role isolation across three purpose-built web portals:

1. **Employee Portal (`/employee/*`)**: Mobile-first portal optimized for rapid field and on-premise check-ins. Provides a single dynamic check-in/check-out button, real-time geofence proximity alerts prior to submission, daily hours/overtime breakdown, and a correction request workflow.
2. **Manager Portal (`/manager/*`)**: Team-scoped supervision console for department leads and supervisors. Automatically restricted server-side to the manager's supervisees. Features real-time team status (Checked In, Late, Not Checked In), team history logs with sanitized coordinates, and a pending correction approval workflow.
3. **Admin Console (`/admin/*`)**: Comprehensive human resources and operations headquarters. Enables multi-site geofence registration, shift rule definition, employee lifecycle onboarding with biometric consent, company-wide attendance auditing, flagged anomaly resolution, and payroll-ready CSV exports.

---

## 🌟 Key Enterprise Capabilities

### 1. Multi-Site Geofencing Engine (`core/geofence.py`)
- **Haversine Spherical Distance**: Calculates high-precision distance in meters between user GPS coordinates and the assigned site anchor point.
- **Configurable Boundary Radius**: Office sites enforce strict radial perimeters (e.g., 200m or 250m). Check-in attempts beyond the radius are captured, geotagged, and flagged for supervisor review rather than silently accepted or dropped.
- **Field & Remote Exemption Policy**: Sites marked with `geofencing_enabled = 0` (e.g., Pan-India Field Operations or Remote Sales) bypass distance rejection while permanently recording GPS coordinates and reverse-geocoded street addresses for auditability.
- **Pre-Submission Warning**: The employee camera UI computes distance client-side prior to capture, warning workers if they are outside office bounds (e.g., *"You appear to be 850m from your registered office — this will be flagged for review"*).

### 2. Industrial Shift Scheduling & Punctuality Engine (`core/shift_engine.py`)
- **Configurable Shifts**: Supports arbitrary shift schedules (e.g., General 09:00–18:00, Early 07:00–16:00, Night 22:00–06:00, Flexible Remote).
- **Grace Period Evaluation**: Check-ins within the grace period (e.g. 15 minutes) are recorded as `on_time`. Arrivals after the grace threshold are automatically classified as `late` with exact minutes past scheduled start.
- **Early Departure & Overtime**: Check-outs before shift completion are flagged as `early_leave`. Work extending beyond scheduled shift hours is classified as `overtime` with net hours calculated.
- **Mandatory Break Deductions**: Deducts scheduled lunch/rest periods (e.g., 60 minutes) only when gross worked time exceeds the break duration.
- **Edge-Case Resilience**: Evaluates inverted timestamps (check-out earlier than check-in), missing check-outs, and overnight shifts crossing midnight.

### 3. Biometric Verification & Liveness Proof
- **128-D Deep Metric Embeddings**: In-memory vector cache with cosine/Euclidean distance matching against enrolled reference angles.
- **Interactive Multi-Frame Liveness**: Eye Aspect Ratio (EAR) blink detection over burst capture frames prevents spoofing with printed photographs, digital screens, or looped video.
- **Anti-Proxy Identity Matching**: Cross-references the authenticated session ID with the recognized facial embedding. If an employee attempts to check in using a colleague's face, the system flags the attempt and returns HTTP 403 Forbidden.
- **Forensic Watermark Stamping (`core/geotag.py`)**: Geocodes GPS coordinates into street addresses via OpenStreetMap Nominatim and burns an unalterable forensic watermark badge (name, employee code, timestamp, site name, distance, geofence status) into proof photos stored in `attendance_photos/`.

### 4. Regulatory Compliance & Biometric Scaffolding
- **Explicit Biometric Consent**: Onboarding captures affirmative biometric consent with an immutable UTC timestamp (`biometric_consent = 1`, `biometric_consent_timestamp = ISO-8601`) before face encoding is permitted (supporting GDPR Art. 9, DPDP Act, and BIPA technical prerequisites).
- **Data Retention & Right to Erasure**: Admin interface provides a 1-click **Purge Biometrics** action upon employee offboarding, permanently destroying raw reference portraits from `dataset/{id}/` and deleting embeddings from the database.
- **Data Minimization & Coordinate Obfuscation**: Managers can view attendance status and city/neighborhood locations, but precise lat/long coordinates and raw biometric images are strictly restricted to administrators and the individual employee.
- **Tamper-Evident Audit Logging (`audit_logs`)**: Centralized logging tracks administrative logins, site creations, shift modifications, employee enrollments, biometric purges, and payroll exports.

---

## 🗄️ Database Domain Model

```
sites
├── id (INTEGER PK)
├── name (TEXT)
├── address (TEXT)
├── latitude (REAL)
├── longitude (REAL)
├── geofence_radius_meters (REAL DEFAULT 200.0)
├── geofencing_enabled (INTEGER DEFAULT 1)
└── created_at (TEXT)

shifts
├── id (INTEGER PK)
├── name (TEXT)
├── start_time (TEXT, 'HH:MM')
├── end_time (TEXT, 'HH:MM')
├── grace_period_minutes (INTEGER DEFAULT 15)
├── break_duration_minutes (INTEGER DEFAULT 60)
└── created_at (TEXT)

departments
├── id (INTEGER PK)
├── name (TEXT)
├── manager_id (INTEGER FK -> employees.id)
└── created_at (TEXT)

employees
├── id (INTEGER PK)
├── name (TEXT)
├── employee_code (TEXT UNIQUE)
├── department_id (INTEGER FK -> departments.id)
├── site_id (INTEGER FK -> sites.id)
├── shift_id (INTEGER FK -> shifts.id)
├── manager_id (INTEGER FK -> employees.id)
├── role (TEXT: 'employee', 'manager', 'admin')
├── email (TEXT UNIQUE)
├── password_hash (TEXT)
├── active (INTEGER DEFAULT 1)
├── biometric_consent (INTEGER DEFAULT 0)
├── biometric_consent_timestamp (TEXT)
└── created_at (TEXT)

attendance_events
├── id (INTEGER PK)
├── employee_id (INTEGER FK -> employees.id)
├── event_type (TEXT: 'check_in', 'check_out')
├── timestamp (TEXT ISO-8601)
├── latitude (REAL)
├── longitude (REAL)
├── address (TEXT)
├── site_id (INTEGER FK -> sites.id)
├── distance_from_site_meters (REAL)
├── within_geofence (INTEGER DEFAULT 1)
├── confidence (REAL)
├── liveness_passed (INTEGER DEFAULT 1)
├── geotagged_photo_path (TEXT)
├── status (TEXT: 'on_time', 'late', 'early_leave', 'overtime', 'flagged')
├── flagged (INTEGER DEFAULT 0)
├── flag_reason (TEXT)
├── reviewed_by (INTEGER FK -> employees.id)
├── review_note (TEXT)
└── created_at (TEXT)

correction_requests
├── id (INTEGER PK)
├── employee_id (INTEGER FK -> employees.id)
├── attendance_date (TEXT 'YYYY-MM-DD')
├── reason (TEXT)
├── requested_change (TEXT)
├── status (TEXT: 'pending', 'approved', 'rejected')
├── reviewed_by (INTEGER FK -> employees.id)
├── review_note (TEXT)
├── reviewed_at (TEXT)
└── created_at (TEXT)

audit_logs
├── id (INTEGER PK)
├── actor_id (INTEGER)
├── actor_role (TEXT)
├── action (TEXT)
├── resource_type (TEXT)
├── resource_id (TEXT/INT)
├── details (TEXT)
└── created_at (TEXT)
```

---

## 📁 Project Directory Structure

```
enterprise-attendance-system/
├── app.py                          # Blueprint registrations, root routing, protected media delivery
├── config.py                       # Application settings, geocoding timeouts, threshold configs
├── requirements.txt                # Python dependencies
├── attendance.db                   # SQLite WAL-mode enterprise database
├── core/
│   ├── auth.py                     # Multi-role decorators & server-side manager scoping
│   ├── db.py                       # Database migrations, default seeders, audit logger
│   ├── face_engine.py              # ResNet-128 metric embeddings & vector similarity cache
│   ├── geofence.py                 # Haversine distance calculator & site boundary validator
│   ├── geotag.py                   # Reverse geocoding & Pillow watermark badge stamper
│   ├── liveness.py                 # Multi-frame EAR blink liveness detector
│   ├── pipeline.py                 # 9-stage event pipeline (Check-in / Check-out)
│   └── shift_engine.py             # Industrial shift evaluation, grace periods, & overtime
├── blueprints/
│   ├── employee/
│   │   ├── __init__.py
│   │   └── routes.py               # Employee camera toggle, history, correction requests
│   ├── manager/
│   │   ├── __init__.py
│   │   └── routes.py               # Team-scoped dashboard, logs, & correction queue
│   ├── admin/
│   │   ├── __init__.py
│   │   └── routes.py               # Sites, shifts, employee roster, review queue, payroll CSV
│   └── portal/
│       ├── __init__.py
│       └── routes.py               # Legacy compatibility blueprint
├── migrations/
│   ├── 001_initial_schema.sql
│   ├── 002_add_roles_and_review_fields.sql
│   └── 003_add_sites_shifts_departments.sql
├── dataset/                        # Enrolled face images partitioned by employee ID
├── attendance_photos/              # Forensic watermarked proof-of-attendance photos
├── scripts/
│   └── seed_enterprise_data.py    # Multi-site enterprise seeder (sites, shifts, employees, cycles)
├── static/
│   ├── css/style.css
│   └── js/
│       ├── employee/camera_checkin.js
│       └── admin/dashboard.js
├── templates/
│   ├── employee/
│   │   ├── base_employee.html
│   │   ├── login.html
│   │   ├── check_in_out.html
│   │   └── my_attendance.html
│   ├── manager/
│   │   ├── base_manager.html
│   │   ├── dashboard.html
│   │   ├── attendance_records.html
│   │   └── correction_requests.html
│   └── admin/
│       ├── base_admin.html
│       ├── login.html
│       ├── dashboard.html
│       ├── sites.html
│       ├── shifts.html
│       ├── employees.html
│       ├── attendance_records.html
│       ├── flagged_queue.html
│       ├── reports.html
│       └── audit_logs.html
└── tests/
    ├── test_geofence.py             # 7 unit tests (boundary radius, haversine, remote bypass)
    ├── test_shift_engine.py         # 8 unit tests (grace period, overtime, lunch break deduction)
    ├── test_auth_boundaries.py      # 8 unit tests (manager team scoping, proxy defense, RBAC)
    ├── test_face_engine.py          # Vector embedding matching & threshold tests
    ├── test_liveness.py             # Blink EAR liveness tests
    ├── test_geotag.py               # Forensic watermark generation tests
    └── test_pipeline_e2e.py         # End-to-end multi-role pipeline tests
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
```bash
# Clone repository
cd Digital-Facial-Recognisation-Attendance-System-main-main

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Seed Enterprise Demo Data
Run the enterprise seeding script to populate sites, shifts, departments, managers, employees, and full-day attendance cycles:
```bash
python scripts/seed_enterprise_data.py
```

### 3. Run Application Server
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## 👥 Demo Credentials for Testing

| Role | Username / Identifier | Password | Managed Entity / Site |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin` | `admin123` | Company-wide governance across all sites |
| **Manager 1** | `vikram.sharma@enterprise.internal` | `password123` | Engineering Department (Aarav, Aditya) |
| **Manager 2** | `priya.patel@enterprise.internal` | `password123` | Operations & Field Sales (Rohan, Sneha) |
| **Employee 1** | `aarav@enterprise.internal` | `password123` | Mumbai Corporate HQ (General Shift) |
| **Employee 2** | `rohan@enterprise.internal` | `password123` | Bengaluru Tech Hub (General Shift) |
| **Employee 3** | `sneha@enterprise.internal` | `password123` | Remote Field Operations (Flexible Shift) |

---

## 📊 Payroll Export Specification (`GET /admin/payroll-export/csv`)

The administrative payroll export generates a standardized, payroll-engine-compatible CSV formatted as follows:

| Column Header | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `Employee ID` | String | Unique company employee identifier | `EMP-001` |
| `Name` | String | Full legal name of employee | `Aarav Sharma` |
| `Department` | String | Assigned organizational department | `Engineering & Architecture` |
| `Site` | String | Assigned primary work location | `Mumbai Corporate HQ` |
| `Shift` | String | Assigned shift rule | `Standard General Shift` |
| `Days Present` | Integer | Total days with verified check-in | `22` |
| `Total Regular Hours` | Float | Scheduled working hours completed | `176.00` |
| `Total Overtime Hours` | Float | Net approved overtime hours accrued | `8.50` |
| `Total Worked Hours` | Float | Sum of regular and overtime hours | `184.50` |
| `Late Check-ins` | Integer | Number of check-ins beyond grace period | `1` |
| `Early Departures` | Integer | Number of check-outs before scheduled end | `0` |
| `Flagged Incidents` | Integer | Number of geofence or proxy violations | `0` |

---

## 🧪 Automated Test Suite

Run all 38 automated test suites across geofencing, shift calculations, role authorization, and pipelines:
```bash
python -m unittest discover tests
```

### Covered Test Categories:
1. **`test_geofence.py` (7 tests)**:
   - Within geofence center calculation.
   - Exact boundary coordinate evaluation ($\le 200\text{m}$).
   - Boundary breach outside registered perimeter.
   - Zero-distance origin point validation.
   - Remote/field bypass policy (`geofencing_enabled = 0`).
   - Missing/invalid GPS coordinate handling.
   - Cross-country / inter-city distance calculation.

2. **`test_shift_engine.py` (8 tests)**:
   - On-time arrival within 15-minute grace period.
   - Late arrival beyond grace limit with exact minute calculation.
   - On-time shift departure.
   - Early leave departure with minute calculation.
   - Overtime calculation with threshold cutoff.
   - Net worked hours with automatic 60-minute break deduction.
   - Edge case: Missing check-out detection (`missing_checkout`).
   - Edge case: Corrupted sequence where check-out precedes check-in (`invalid_sequence`).

3. **`test_auth_boundaries.py` (8 tests)**:
   - Employee portal isolation (employees cannot see other employees' data).
   - Employee forbidden from accessing manager or admin endpoints.
   - Manager scoped queries (manager cannot view employees from other departments).
   - Manager cannot approve correction requests for employees outside their team.
   - Biometric proxy prevention (detects and blocks face mismatches).
   - Role-based media protection for attendance photos.

---

## ⚖️ Compliance Scaffolding & Legal Disclaimer

> [!WARNING]
> **Legal Disclaimer**: This platform provides technical scaffolding to support compliance with biometric privacy frameworks (such as GDPR Article 9, India's Digital Personal Data Protection Act, and the Illinois Biometric Information Privacy Act). Implementing this software does not constitute legal certification. Before production deployment, the deploying organization must review their configuration with qualified legal and HR counsel for their specific operating jurisdiction.
