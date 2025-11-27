# FireSim Paper Workloads

* Untested as of 08/07/2024 *

Old FireSim software workloads used in the FireSim paper (see below).
This software needs the following to build:

1. Setup FireMarshal, FireSim's workload builder: https://github.com/firesim/FireMarshal. This was last tested with commit `74ac78a`.
2. Ensure you have all FireMarshal requirements on your path (i.e. if you are using Chipyard, ensure the `env.sh` is setup)
3. Symlink the FireMarshal repository to this repository like so:


```
ln -sf <PATH_TO_FIREMARSHAL> firemarshal-symlink
```

4. Build using `make allpaper` or `make linux-poweroff`

### **ISCA 2018**: FireSim: FPGA-Accelerated Cycle-Exact Scale-Out System Simulation in the Public Cloud

You can learn more about FireSim in our ISCA 2018 paper, which covers the overall FireSim infrastructure and large distributed simulations of networked clusters. This paper was selected as one of **IEEE Micro’s "Top Picks from Computer Architecture Conferences, 2018"** and for **"ISCA@50 25-year Retrospective 1996-2020"**.

> Sagar Karandikar, Howard Mao, Donggyu Kim, David Biancolin, Alon Amid, Dayeol
Lee, Nathan Pemberton, Emmanuel Amaro, Colin Schmidt, Aditya Chopra, Qijing
Huang, Kyle Kovacs, Borivoje Nikolic, Randy Katz, Jonathan Bachrach, and Krste
Asanović. **FireSim: FPGA-Accelerated Cycle-Exact Scale-Out System Simulation in
the Public Cloud**. *In proceedings of the 45th International Symposium
on Computer Architecture (ISCA’18)*, Los Angeles, CA, June 2018.

[Paper PDF](https://sagark.org/assets/pubs/firesim-isca2018.pdf) | [IEEE Xplore](https://ieeexplore.ieee.org/document/8416816) | [ACM DL](https://dl.acm.org/citation.cfm?id=3276543) | [BibTeX](https://sagark.org/assets/pubs/firesim-isca2018.bib.txt)
