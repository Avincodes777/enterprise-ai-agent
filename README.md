# Enterprise Knowledge & Action Agent

## 1. What the Project Is
The **Enterprise Knowledge & Action Agent** is a production-grade enterprise AI system designed to intelligently connect organizational knowledge sources, structured databases, and external enterprise tools. When fully built, the platform will enable automated knowledge retrieval, multi-step agent reasoning, human-in-the-loop action workflows, and verified execution across enterprise systems.

---

## 2. Current Day 1 Status
We are currently on **Day 1: Project Foundation Setup**. 
- Initial clean folder structure established.
- Minimal FastAPI backend created with a baseline `/health` check endpoint.
- Minimal React + Vite frontend scaffolded.
- Environment variable template (`.env.example`) and `.gitignore` defined.
- Foundational `docker-compose.yml` created.
- **No AI, RAG, agents, or external database integrations have been implemented yet.**

---

## 3. Technologies Currently in Use
- **Backend:** Python 3.11+, FastAPI, Uvicorn
- **Frontend:** React 18, Vite
- **Containerization:** Docker, Docker Compose
- **Configuration:** Environment variables (`.env`)

---

## 4. Future Planned Architecture
In upcoming phases, the project will expand into an enterprise-grade AI architecture comprising:
- **Agent Orchestration:** LangGraph stateful multi-agent workflows
- **Retrieval-Augmented Generation (RAG):** Advanced hybrid search and document chunking pipelines
- **Vector Database:** Qdrant for semantic search and embeddings
- **Relational Database & Cache:** PostgreSQL (operational data & state) and Redis (caching & memory)
- **Tool Integrations:** SQL Agents and connectors (e.g., Slack, GitHub, Notion, Google Drive)
- **Security & Governance:** RBAC (Role-Based Access Control), auth layers, permission sandboxing, and human-in-the-loop verification
- **Evaluation & Observability:** Systematic evaluation pipelines and tracing
- **Deployment & Cloud:** AWS infrastructure, container orchestration, and CI/CD pipelines
