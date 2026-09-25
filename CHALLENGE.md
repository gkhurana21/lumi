# Live Character Robot Software Challenge

## Objective

Build a live, expressive character around the supplied fictional five-degree-of-
freedom lamp robot.

![Dummy five-DOF lamp robot](robot/dummy-lamp.png)

For this exercise, the character uses the laptop camera to see, the laptop
microphone to listen, and the laptop speaker for voice, music, and sound
effects. The supplied robot model provides the simulated body for motion and
light.

The result should feel like one aware character, not a collection of unrelated
AI demonstrations.

## Demonstration

Demonstrate one continuous interaction containing these moments:

1. **Engagement** — a person looks toward the character and the character
   recognizes the opportunity to interact. It disengages appropriately when
   attention moves elsewhere.
2. **Character response** — it acknowledges the person with an intentional
   combination of motion, light, voice, sound effects, or music.
3. **Spoken interaction** — it listens through the laptop microphone and
   responds through the laptop speaker.
4. **Scene memory** — it observes at least one object through the laptop camera,
   retains useful information about it, and later answers a spoken question
   about that object.
5. **Goal-directed action** — it receives a spoken goal referring to the live
   scene, uses vision and language to choose a sequence of lamp actions, and
   observes the scene again before completing the goal.

Across the complete demonstration, show purposeful use of motion, light,
voice, a sound effect, and music. They do not need to occur simultaneously.
The goal-directed action may be combined with any of the other moments.

For this challenge, VLA means that live visual evidence and a language goal
influence robot action. You may use a VLA model or compose vision, language, and
control components. Define the action representation and the boundary between
model decisions and body execution.

## Technical freedom

You may choose:

- Programming languages and frameworks.
- Process and repository structure.
- Simulation or visualization technology.
- Communication between the character software and simulated robot body.
- Perception, speech, language, and memory approaches.
- Local models, remote services, or a hybrid approach.
- Packaging and deployment method.

Perception may use cloud APIs, including visual understanding and speech
services. Training or running perception models locally is not required. If
cloud services are used, describe what data is sent, where decisions are made,
and why that approach fits the system.

Explain the important choices and their tradeoffs. The supplied URDF defines a
robot model, not a required software architecture.

## Target environment

The intended deployment target is a clean Ubuntu 24.04 LTS laptop with:

- Four CPU cores.
- 8 GB of RAM.
- No discrete GPU or CUDA requirement.
- An integrated or USB camera.
- A microphone and speaker exposed through standard Linux audio facilities.
- Wi-Fi connectivity.

Development and the demonstration may use another operating system,
but the submitted deployment approach must address the target above.

## Timebox

Spend no more than **6–8 hours** on the exercise. Prefer a smaller, coherent
system with clear decisions over a wide but unfinished implementation. State
what you completed and what you intentionally left out.

## Supplied files

- [`robot/dummy_lamp_5dof.urdf`](robot/dummy_lamp_5dof.urdf)
- [`robot/dummy-lamp.png`](robot/dummy-lamp.png)
- [`robot/assets/lamp_shade.stl`](robot/assets/lamp_shade.stl)
- [`SUBMISSION.md`](SUBMISSION.md)

No simulator, runtime scaffold, protocol, model wrapper, service template, or
reference implementation is supplied.

## Use of AI tools

AI-assisted development tools are allowed. You remain responsible for the
architecture, behavior, measurements, code, and technical decisions in the
submission, and should be able to explain and modify the work.
