---
title: "QR Nativity Challenge"
date: 2024-12-01
summary: "Gamified QR-based Christmas challenge for local commerce."
tags: [ "software", "web" ]
tech: [ "Angular", "Django REST API", "Google OAuth", "QR" ]
cover:
  image: "/projects/repte_pessebre.png"
repo: "https://github.com/Pessebres-del-Segre/repte"
demo: "https://pessebres-del-segre.github.io/repte"
---

### Problem

The local Nativity Scene Contest (Concurs de Pessebres) in Artesa de Segre needed stronger **collaboration with local
shops** to increase visibility and participation. The baseline was **14 participating shops** the previous year, and the
goal was to grow that number by creating a more compelling, town-wide experience that benefits both the association and
the local economy.

### Constraints

- **Low-friction participation**: the experience had to work on any phone, with no app-store install required.
- **Real-world deployment**: QR codes and physical dioramas needed to be placed in shop windows and remain usable
  throughout the campaign.
- **Fairness and maintainability**: shops needed a consistent way to participate while keeping the initiative scalable
  year-over-year.
- **Cultural authenticity**: the content had to reflect Artesa’s identity (local traits, references, songs), not a
  generic Christmas storyline.
- **Multi-stakeholder coordination**: align an association, shop owners, and the local high school—each with different
  incentives and constraints.

### My role

I acted as the **main organizer** and **product owner** of the initiative, defining the concept, rules, and rollout
strategy. I
was also responsible for the **technical design** and development of the platform, including system architecture and
implementation. In addition, I designed the user experience and visual identity of the challenge and 
**led its promotion**
through appearances on local television and in local newspapers to explain the project and encourage participation.

### Architecture

At a high level, the system combines a QR-driven user journey with a lightweight web platform:

#### **User journey (mobile-first)**

Visitors walk through the town and **scan QR codes** displayed in the windows of participating shops. Each scan unlocks
a
**fragment of an original story** that narrates the Nativity of Artesa de Segre, incorporating local cultural traits and
traditional songs. The experience is progressive and encourages movement across different areas of the town,
transforming the contest into an exploratory, town-wide activity rather than a single-location event.

Beyond the digital story, a key component of the project was the creation of **unique physical nativity scenes (
diorames)**
for each participating shop. These were developed in collaboration with the local high school, which strengthened ties
between the association and younger participants, some of whom are between 8 and 16 years old. This collaboration not
only increased the number and diversity of dioramas, but also **positioned students as active contributors** to a
cultural
initiative within their own town.

#### **Core components**

- **Frontend**: Angular web app (mobile-friendly) for scanning, progress tracking, and story reading. :
  contentReference
- **Backend**: Django REST API for users, progress, story chapters, and shop content management. :
  contentReference
- **Auth**: OAuth-based login to reduce friction and support secure administration.
- **Shop portal**: a dedicated panel where each shop can upload and maintain its description (and associated content),
  keeping operations decentralized and sustainable.


### Results / metrics

- **Local business engagement doubled**: we achieved a **2× increase in the number of local shops interested in
  promoting the contest** (from the prior baseline of 14 shops to significantly more, 30).
- **Stronger town-wide visibility**: by embedding the challenge into the Christmas program and shop windows, the contest
  shifted from a “single-event” dynamic to a **distributed, repeatable experience** across the holiday period. :
  contentReference

### Lessons learned


- **Gamification works best when it is aligned with real incentives**: tying story completion to visiting (and
  supporting) shops created a clearer value exchange than “scan just for points.”
- **Operational tooling matters as much as the app**: the shop panel reduced bottlenecks and made participation
  maintainable without constant central coordination.
- **Physical + digital (“phygital”) beats digital-only for local culture**: the dioramas turned QR scanning into a
  meaningful on-street experience, not just a link to content.
- **Youth partnerships are leverage**: involving the high school improved both production capacity (more dioramas) and
  community buy-in.

Blending physical experiences with lightweight digital interactions proved far more effective than a purely online
solution for a local cultural event. Giving shops autonomy through simple management tools was essential for scalability
and long-term sustainability. Finally, involving younger participants as co-creators, rather than passive attendees,
significantly improved community engagement and strengthened the social impact of the project.

