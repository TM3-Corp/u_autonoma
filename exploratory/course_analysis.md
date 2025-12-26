● The Key Problem
Good instructional design ≠ Good prediction potential
┌─────────────────────────────────────────────────────────────────┐
│ COURSE ANALYSIS │
├─────────────────────────────────┬───────────────────────────────┤
│ INSTRUCTIONAL DESIGN │ PREDICTION POTENTIAL │
│ (What's in the course?) │ (Can we build a model?) │
├─────────────────────────────────┼───────────────────────────────┤
│ • Modules (structure) │ • Grades IN Canvas │
│ • Assignments (activities) │ • Grade variance > 15% │
│ • Quizzes (formative) │ • Failure rate 20-80% │
│ • Pages (content) │ • N students > 15 │
│ • Files (materials) │ • Activity data exists │
│ • Discussions (interaction) │ │
├─────────────────────────────────┼───────────────────────────────┤
│ Endpoints: │ Endpoints: │
│ /modules │ /enrollments?include=grades │
│ /assignments │ /students/submissions │
│ /quizzes │ /analytics/student_summaries │
│ /pages │ │
│ /files │ │
│ /discussion_topics │ │
└─────────────────────────────────┴───────────────────────────────┘
The disconnect:

- A course can have Excelente design (28 modules, 55 assignments) but NO grades in Canvas (uses "Libro de Calificaciones")
- A course can have Básico design (3 modules, 5 assignments) but valid grades we can analyze
  What We Found in Control de Gestión
  | Course | Design | Modules | Assignments | Has Canvas Grades? |
  |---------------------------------|-----------|---------|-------------|--------------------|
  | GESTIÓN DEL TALENTO-P01 | Excelente | 28 | 55 | :x: No |
  | INGLÉS II-P01 | Excelente | 38 | 36 | :x: No |
  | FUND BUSINESS ANALYTICS-P01 | Bueno | ~10 | ~15 | :marca_de_verificación_blanca: Yes |
  | TALL COMPETENCIAS DIGITALES-P01 | Bueno | ~8 | ~12 | :marca_de_verificación_blanca: Yes |
  Only 5 of 35 active courses have grades IN Canvas - the rest use the external gradebook.
  The Endpoints We Use
  For Design Quality (Radiografía):

  /api/v1/courses/{id}/modules # Structure
  /api/v1/courses/{id}/assignments # Activities
  /api/v1/courses/{id}/quizzes # Assessments
  /api/v1/courses/{id}/pages # Content
  /api/v1/courses/{id}/files # Materials
  /api/v1/courses/{id}/discussion_topics # Interaction
  For Prediction Potential:
  /api/v1/courses/{id}/enrollments?include[]=grades # final_score

  /api/v1/courses/{id}/analytics/activity
  /api/v1/courses/{id}/analytics/users/{user_id}/activity

# Then check: N > 10, StdDev > 15, fail_rate 10-90%
