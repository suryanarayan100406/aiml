"""
Generate synthetic task descriptions for NLP model training.
Creates a labeled CSV with columns: text, label, cognitive_demand

Labels:
  0: DEEP_WORK
  1: SHALLOW_WORK
  2: CREATIVE
  3: ADMINISTRATIVE
  4: COMMUNICATION
"""

import csv
import random
import os

# ─── Task Templates Per Category ───────────────────────────────────────────────

DEEP_WORK_TEMPLATES = [
    "Implement {algo} algorithm for the {component} module",
    "Debug the {issue} crash in the {component} service",
    "Refactor the {component} codebase to use {pattern} pattern",
    "Write unit tests for the {component} engine covering edge cases",
    "Optimize {component} query performance reducing latency by 50%",
    "Design database schema for {feature} with normalization",
    "Analyze {data_type} dataset and build regression model",
    "Write technical specification for {feature} architecture",
    "Implement {protocol} authentication flow with token refresh",
    "Build data pipeline for processing {data_type} in real-time",
    "Migrate {component} from monolith to microservice architecture",
    "Create machine learning feature extraction for {data_type} analysis",
    "Write compiler pass for {algo} optimization in the build system",
    "Implement distributed {algo} consensus protocol for {component}",
    "Design and implement caching strategy for {component} reducing DB load",
    "Profile and fix memory leak in {component} under high concurrency",
    "Build real-time {data_type} processing pipeline with exactly-once semantics",
    "Implement custom {algo} solver for the constraint optimization engine",
    "Write formal verification proofs for the {component} safety module",
    "Design fault-tolerant {component} with automatic failover and recovery",
]

SHALLOW_WORK_TEMPLATES = [
    "Update {component} dependency versions in package.json",
    "Fix typo in {doc} documentation page",
    "Add logging to {component} endpoint",
    "Update README with new {feature} setup instructions",
    "Rename {old_name} variable to {new_name} across codebase",
    "Add input validation for {field} field in {component} form",
    "Update {config} configuration for staging environment",
    "Move {file} to the {component} directory",
    "Add environment variable for {config} setting",
    "Run linter and fix formatting issues in {component}",
    "Update API version number to {version}",
    "Add missing type annotations to {component} module",
    "Clean up unused imports in {component} files",
    "Update changelog for version {version} release",
    "Pin {dependency} to specific version for stability",
    "Add default value for {field} in {component} model",
    "Update CI pipeline to use Node {version}",
    "Fix broken link in {doc} documentation",
    "Add .env.example file with required variables",
    "Bump version number for {component} hotfix release",
]

CREATIVE_TEMPLATES = [
    "Design new onboarding flow for first-time {user_type} users",
    "Create visual identity for {brand} product launch",
    "Brainstorm innovative solutions for {problem} user pain point",
    "Design interactive {component} visualization with animations",
    "Write compelling copy for {page} landing page",
    "Prototype new {feature} experience using Figma",
    "Create motion design for {component} state transitions",
    "Design gamification system for {feature} user engagement",
    "Sketch wireframes for {feature} mobile experience",
    "Create illustration set for {doc} help center articles",
    "Design data visualization dashboard for {data_type} metrics",
    "Compose original background music for {feature} meditation mode",
    "Create brand storytelling narrative for {brand} campaign",
    "Design micro-interactions for {component} hover and focus states",
    "Build generative art system for user profile avatars",
    "Create typography system for {brand} design language",
    "Design immersive {feature} experience with parallax scrolling",
    "Storyboard tutorial video for {feature} walkthrough",
    "Create responsive illustration that adapts to {component} viewport",
    "Design award-worthy UI for {feature} settings panel",
]

ADMINISTRATIVE_TEMPLATES = [
    "Review and approve {count} pending pull requests",
    "Update {doc} JIRA tickets with current sprint status",
    "Organize team standup notes from this week",
    "Schedule {meeting_type} meeting with {team} team",
    "Process expense reports for {month} purchases",
    "Update project timeline in {tool} for Q{quarter} milestones",
    "File quarterly {report_type} compliance report",
    "Review and update team access permissions in {tool}",
    "Create onboarding checklist for new {role} hire",
    "Audit {component} service uptime logs for last month",
    "Prepare slide deck for {meeting_type} stakeholder presentation",
    "Update team roster and contact information in HR system",
    "Review and categorize incoming support tickets for triage",
    "Reconcile {month} budget allocation across departments",
    "Document standard operating procedures for {process} workflow",
    "Archive completed {component} project files and close tickets",
    "Compile weekly status report for {team} management review",
    "Coordinate vendor contract renewal for {tool} licenses",
    "Update inventory of development hardware and software assets",
    "Plan and book travel for upcoming {meeting_type} conference",
]

COMMUNICATION_TEMPLATES = [
    "Draft email to {team} team about {topic} deadline change",
    "Prepare presentation for {meeting_type} quarterly review",
    "Write blog post about our {feature} technical architecture",
    "Reply to client feedback about {component} performance issues",
    "Create internal FAQ document for {feature} rollout",
    "Record demo video showing {feature} new capabilities",
    "Write release notes for {component} version {version}",
    "Draft proposal for {feature} partnership opportunity",
    "Compose newsletter update about {topic} progress this quarter",
    "Create tutorial walkthrough for {feature} API integration",
    "Write incident postmortem for the {component} outage last week",
    "Prepare talking points for {meeting_type} customer call",
    "Draft SOW document for {feature} consulting engagement",
    "Write technical blog comparing {algo} vs alternative approaches",
    "Create onboarding documentation for {component} SDK users",
    "Record podcast episode discussing {topic} industry trends",
    "Draft press release for {feature} product announcement",
    "Write RFP response for {component} enterprise contract",
    "Create knowledge base article for {feature} troubleshooting",
    "Compose apology communication regarding {component} service disruption",
]

# ─── Fill-in Values ────────────────────────────────────────────────────────────

FILL_VALUES = {
    "algo": ["binary search", "A*", "gradient descent", "Dijkstra", "quicksort",
             "backpropagation", "dynamic programming", "BFS", "Monte Carlo",
             "simulated annealing", "genetic", "k-means", "random forest"],
    "component": ["payment", "auth", "search", "notification", "analytics",
                  "dashboard", "user-profile", "inventory", "messaging",
                  "billing", "scheduling", "reporting", "cache", "gateway"],
    "issue": ["null pointer", "race condition", "memory leak", "timeout",
             "deadlock", "stack overflow", "segfault", "OOM"],
    "pattern": ["observer", "strategy", "factory", "singleton", "decorator",
               "repository", "CQRS", "event-driven", "hexagonal"],
    "feature": ["dark mode", "real-time sync", "multi-tenant", "offline-first",
               "push notification", "two-factor auth", "auto-save", "undo-redo",
               "collaborative editing", "version history", "export"],
    "data_type": ["time-series", "geospatial", "clickstream", "log",
                 "transaction", "sensor", "genomic", "NLP corpus"],
    "protocol": ["OAuth 2.0", "JWT", "SAML", "OpenID Connect", "mTLS"],
    "doc": ["API reference", "getting started", "deployment", "architecture",
           "contributing", "security", "migration"],
    "old_name": ["userData", "tempVal", "processItem", "handleEvent"],
    "new_name": ["userProfile", "intermediateValue", "transformItem", "onEvent"],
    "field": ["email", "phone_number", "address", "date_of_birth", "username"],
    "config": ["database", "redis", "S3", "CDN", "logging", "feature-flag"],
    "file": ["utils.py", "helpers.js", "constants.ts", "types.d.ts"],
    "version": ["3.2.1", "4.0.0", "2.8.0", "5.1.0", "1.12.0"],
    "dependency": ["lodash", "axios", "moment", "webpack", "prisma"],
    "user_type": ["enterprise", "developer", "student", "creator", "analyst"],
    "brand": ["NovaTech", "Luminary", "AuraSync", "FlowState", "Zenith"],
    "problem": ["onboarding drop-off", "feature discoverability", "retention",
               "mobile performance", "accessibility"],
    "page": ["homepage", "pricing", "product tour", "signup", "features"],
    "count": ["12", "8", "15", "6", "20"],
    "meeting_type": ["sprint planning", "retrospective", "all-hands",
                    "1-on-1", "design review", "architecture", "stakeholder"],
    "team": ["engineering", "product", "design", "QA", "DevOps", "marketing"],
    "month": ["January", "February", "March", "October", "November"],
    "tool": ["Jira", "Confluence", "Notion", "Linear", "Asana", "GitHub"],
    "quarter": ["1", "2", "3", "4"],
    "report_type": ["SOC2", "GDPR", "accessibility", "security", "financial"],
    "role": ["frontend engineer", "backend engineer", "designer", "PM", "QA"],
    "process": ["deployment", "incident response", "code review", "release"],
    "topic": ["Q1 roadmap", "infrastructure migration", "team restructuring",
             "product launch", "security audit", "performance optimization"],
}


def fill_template(template: str) -> str:
    """Replace {placeholder} tokens with random values."""
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_tasks(num_per_category: int = 200) -> list:
    """Generate labeled task descriptions."""
    categories = [
        (DEEP_WORK_TEMPLATES, 0, "DEEP_WORK"),
        (SHALLOW_WORK_TEMPLATES, 1, "SHALLOW_WORK"),
        (CREATIVE_TEMPLATES, 2, "CREATIVE"),
        (ADMINISTRATIVE_TEMPLATES, 3, "ADMINISTRATIVE"),
        (COMMUNICATION_TEMPLATES, 4, "COMMUNICATION"),
    ]

    # Cognitive demand ranges per category (mean, std)
    demand_profiles = {
        0: (0.85, 0.08),   # DEEP_WORK: high demand
        1: (0.25, 0.10),   # SHALLOW_WORK: low demand
        2: (0.70, 0.12),   # CREATIVE: moderate-high
        3: (0.35, 0.10),   # ADMINISTRATIVE: low-moderate
        4: (0.50, 0.12),   # COMMUNICATION: moderate
    }

    tasks = []
    for templates, label, label_name in categories:
        for _ in range(num_per_category):
            template = random.choice(templates)
            text = fill_template(template)
            mean, std = demand_profiles[label]
            demand = max(0.0, min(1.0, random.gauss(mean, std)))
            tasks.append({
                "text": text,
                "label": label,
                "label_name": label_name,
                "cognitive_demand": round(demand, 3),
            })

    random.shuffle(tasks)
    return tasks


def main():
    random.seed(42)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "labeled_tasks.csv")

    tasks = generate_tasks(num_per_category=200)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "label_name", "cognitive_demand"])
        writer.writeheader()
        writer.writerows(tasks)

    print(f"✅ Generated {len(tasks)} labeled tasks → {output_path}")

    # Print distribution
    from collections import Counter
    dist = Counter(t["label_name"] for t in tasks)
    for label, count in sorted(dist.items()):
        print(f"   {label}: {count}")


if __name__ == "__main__":
    main()
