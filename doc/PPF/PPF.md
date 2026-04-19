# P302 - Project Proposal Form

| <br />               | <br />            |
| :------------------- | :---------------- |
| **Learner Name**     | \[Learner number] |
| **Centre Name**      | <br />            |
| **Centre Number**    | <br />            |
| **Teacher Assessor** | <br />            |
| **Date**             | <br />            |
| **Unit**             | P302              |

## Proposed Project Title

*Theoretical and Simulated Dynamics of a Damped Double Pendulum under Small Angles*

***

## Section One: Title, Objectives, Responsibilities

### Title or working title of project

*Theoretical and Simulated Dynamics of a Damped Double Pendulum under Small Angles*

> <br />

### Project objectives

*(e.g., what is the question you want to answer? What do you want to learn how to do? What do you want to find out?)*

> want to learn how to derivation double pendulum' s dynamic equations, by learning Lagrangian method. I want to use experiment and stimulation to prove the theory of double pendulum(angular frequency) under small angle vibration. So I need learn how to use the Lagrangian method to systematically derive the dynamic equations of the double pendulum under small angle conditions, grasp  the key steps and application points of the Lagrangian method in solving mechanical vibration problems.&#x20;
>
> In addition, I need to learn how to design and carry out the double pendulum small angle vibration experiment, correctly use experimental tools to measure the angular frequency and related data, and learn to use ai coding tools (trea) to help me write stimulation program 
>
> I really want link my project with ai, learnig  how to use ai coding tools (trea) to help me write stimulation program.

*(If it is a group project, what will your role or responsibilities be?)*

> <br />

***

## Section Two: Reasons for choosing this project

*(e.g., links to other subjects you are studying, personal interest, future plans, knowledge/skills you want to improve, reasons why the topic is important)*

> After completing A-Level M1 and M2, I found that Newtonian force analysis becomes impractical for multi-object systems. The simple pendulum and projectile motion with air resistance were manageable, but adding a second pendulum creates coupled equations that are extremely messy to derive via free-body diagrams.  I then learned ordinary differential equations in FP2 and P4. Recognising that ODEs are not merely a mathematical exercise, I applied them to the simple pendulum (second‑order linear ODE) and to projectile motion with linear drag. These practices confirmed that differential equations allow systematic handling of problems where forces vary with position or velocity.  The double pendulum under small‑angle approximation is a natural next step. Its two degrees of freedom lead to a pair of coupled linear ODEs – a classic system for studying normal modes. Using Lagrangian mechanics instead of force analysis avoids tedious constraint forces and yields the equations cleanly. This model is neither trivial (like a single pendulum) nor too advanced (like a continuous system). It bridges A‑Level further maths to university analytical mechanics.  Working on this EPQ also prepares me for engineering or physics degrees, where coupled oscillators and variational principles appear repeatedly. Solving the double pendulum solidifies my ability to translate physical systems into differential equations and interpret their solutions.

***

## Section Three: Activities and timescales

| Activities to be carried out during the project *(e.g., research, data collection, numerical analysis, writing, preparing for the presentation, etc.)*                                            | How long this will take |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------- |
| **Theoretical Research & Derivation**: Study Lagrangian mechanics, derive the dynamic equations for the double pendulum under small-angle approximation, and find the normal modes (frequencies). | 1.5 weeks               |
| **Computer Simulation**: Write Python simulation code for the small-angle double pendulum using Trae IDE, and visualize the theoretical results (matplotlib).                                     | 1 week                  |
| **Experiment Setup & Data Collection**: Conduct single pendulum damping tests, perform double pendulum experiments, record videos, and extract trajectory data using Tracker.                     | 1.5 weeks               |
| **Data Analysis**: Compare experimental data (angular frequencies) with theoretical predictions and simulation results. Analyze errors and damping effects.                                       | 1 week                  |
| **Report Writing (First Draft)**: Write the 6000-word dissertation in LaTeX, covering introduction, theory, simulation, experimental design, results, and conclusion.                             | 2 weeks                 |
| **Finalisation & Presentation**: Refine the final essay, complete the Activity Log and PPF, finalize the GitHub repository, and prepare the presentation PPT.                                     | 1 week                  |
| **Milestone one: First Draft Submission** (6000 words, basic formatting met)                                                                                                                      | <br />                  |
| Target date *(set by tutor-assessor)*:                                                                                                                                                            | 17th April              |
| **Milestone two: Final Submission** (Final essay, Activity Log, PPF, GitHub repo, PPT)                                                                                                            | <br />                  |
| Target date *(set by tutor-assessor)*:                                                                                                                                                            | 30th April              |

***

## Section Four: Resources

### Required resources

*(What resources will you need for your research, data collection, write up and presentation e.g., equipment, ICT, libraries, books, journals)*

> **oundational Textbooks & Theoretical Resources**\
> To develop a rigorous theoretical foundation, I will rely on graduate-level textbooks that provide a comprehensive treatment of Lagrangian mechanics and small oscillations.
>
> - **Classical Mechanics (3rd Edition)** by Herbert Goldstein, Charles Poole, and John Safko. Specifically, **Chapter 6: "Small Oscillations"** will be my primary guide for formulating the linearized equations of motion, calculating normal mode frequencies, and deriving the normal coordinates for the double pendulum.
> - **Mechanics (Volume 1)** by Landau and Lifshitz. I will consult **Section 23: "Small Oscillations"** for its concise, first-principles approach to the general theory of small oscillations, which offers a valuable complementary perspective to Goldstein.
> - **Classical Mechanics** by John R. Taylor. I will use **Chapter 11: "Coupled Oscillators and Normal Modes"** for its clear, step-by-step derivations and numerous solved examples, which will help me verify my own algebraic derivations.
>
> **2. Simulation & AI-Assisted Coding Tools**\
> To bridge the gap between theory and experiment, I will use computational tools to simulate the system. I plan to leverage AI code editors to expedite the development and debugging of the simulation code.
>
> - **Programming Language & Environment:** Python with the following libraries:
>   - **NumPy/SciPy:** For matrix operations and numerically solving the system of linearized differential equations (ODE solvers like `scipy.integrate.solve_ivp`).
>   - **Matplotlib/Plotly:** To generate high-quality animations and phase-space plots for visual analysis of the small-angle motion.
>   - **SymPy:** To perform the symbolic derivation of the Lagrangian and the subsequent linearization, ensuring my algebra is error-free.
> - **AI Coding Assistant:** I will use an AI code editor (such as GitHub Copilot or Cursor) to assist in:
>   - Rapidly prototyping the simulation script.
>   - Debugging errors in numerical integration.
>   - Generating code for data acquisition and analysis from experimental videos.
>
> **3. Experimental & Data Collection Equipment**\
> To experimentally validate the theoretical and computational results, I will construct a physical double pendulum and acquire motion data.
>
> - **Physical Apparatus:**
>   - Two precisely machined aluminum or steel rods with low-friction bearings at the pivot points to minimize energy loss and ensure the system approximates the idealized model.
>   - A sturdy support stand clamped to a heavy base to eliminate external vibrations.
> - **Data Acquisition:**
>   - A high-speed digital camera (e.g., a modern smartphone capable of 120/240 fps recording or a dedicated USB camera) to capture the motion of the bobs with sufficient temporal resolution for accurate small-angle analysis.
>   - **Video Analysis Software:** Tracker, an open-source physics video analysis tool, will be essential for digitizing the position-time data of the two bobs from the recorded videos. This data will be compared directly with the simulation outputs.
>
> **4. Literature & Visual Learning Resources**\
> To gain an intuitive understanding of the system's behavior and to find established methodologies, I will use online academic and video platforms.
>
> - **Video Platforms (YouTube/Bilibili):** I will use these platforms to study existing experimental setups. Specific channels like "Harvard Natural Sciences Lecture Demonstrations" and "SmarterEveryDay" provide high-quality visualizations of coupled oscillators. I will focus on animations that clearly show the two normal modes (the in-phase and out-of-phase oscillations) to build a qualitative understanding before performing my own experiment.
> - **Academic Databases:** I will use my university’s library portal to access journals such as the *American Journal of Physics*. I will search for articles like "The Double Pendulum: A Classic Example of Normal Modes" to find established experimental techniques and data analysis methods.
>
> **5. Resources for Presentation & Write-up**
>
> - **Document Preparation:** LaTeX (Overleaf) for typesetting the theoretical derivations and final report.
> - **Presentation Software:** Microsoft PowerPoint or Canva to create clear, professional slides for my final presentation, integrating the simulation animations and experimental video clips.

### Areas of research

*(What areas of research will you cover?)*

> 1. **Theoretical Physics & Mathematics:** Application of Lagrangian mechanics to set up differential equations for coupled oscillators, and utilizing linear algebra to find eigenvalues/eigenvectors for normal modes.
> 2. **Computational Physics:** Using Python and numerical integration (ODE solvers) to simulate dynamic systems, along with exploring how AI coding tools can assist beginners in writing functional physics simulations.
> 3. **Experimental Methodology:** Techniques for capturing and analyzing physical motion using video tracking software (Tracker), and methods for identifying damping coefficients.

***

## Section Five: Contingencies

| What problems might you have in the data collection process? | What will you do to try to stop this from happening? What will you do if it does happen? |
| :----------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| **High joint friction ruining the small-angle motion:** The physical double pendulum might lose energy too quickly, making it hard to observe the theoretical normal modes. | **Prevention:** I will use smooth ball bearings and lubricate them before the experiment. <br> **Action if it happens:** I will measure the damping coefficient using a single pendulum first, and incorporate this real damping into my Python simulation to match the messy experimental reality instead of ignoring it. |
| **Video tracking software (Tracker) failing to auto-track:** The camera might blur the fast-moving pendulum bob, causing the software to lose the tracking point. | **Prevention:** I will stick bright, highly contrasting markers (like neon tape) on the pendulum bobs and ensure the room is very well-lit. <br> **Action if it happens:** I will manually click frame-by-frame for the crucial seconds of data, or re-record the video using a higher frame rate (e.g., 120fps on my phone). |
| **AI-generated simulation code failing to run:** The AI tool might suggest incorrect Python syntax or use deprecated SciPy functions, halting my simulation progress. | **Prevention:** I will not blindly copy-paste. I will ask the AI to explain each block of code and verify the ODE solver syntax against the official SciPy documentation. <br> **Action if it happens:** I will use the AI tool's debugging feature to feed the error traceback back into the prompt, or seek help from online forums (StackOverflow) and my computer science teacher. |

***

## Comments and Agreement

### From tutor-assessor

**Comments (optional):**

> <br />

I confirm that the project is appropriate.

- **Agreed (Name):**
- **Date:**

### From proposal checker

**Comments (optional):**

> <br />

I confirm that the project is appropriate.

- **Agreed (Name):**
- **Date:**

