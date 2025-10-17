# Project 'Nexus' - Onboarding Guide

## Local Setup for Project Nexus

This guide will help you set up your local environment for Project Nexus.

**1. Install Git:**

*   Follow the instructions on the [official Git website](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your machine.

**2. Install Docker:**

*   Download and install Docker Desktop from the [official Docker website](https://www.docker.com/products/docker-desktop).

**3. Install Python:**

*   For Project Nexus, you'll need Python 3.9. You can download it from the [official Python website](https://www.python.org/downloads/release/python-390/).

**4. Install Node.js:**

*   We use Node.js version 16. You can download it from the [official Node.js website](https://nodejs.org/en/blog/release/v16.0.0).

**5. Install Proprietary Library:**

*   Install the internal 'Nexus' library by running: `pip install nexus-library`

### Requesting Cloud Resources

To get access to our cloud resources for Project Nexus, use the `CloudProvisioner` tool.

1.  Open your terminal.
2.  Run the following command to request access for the Nexus team:

    ```bash
    cprov request --team=nexus-dev
    ```
