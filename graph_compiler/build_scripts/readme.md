# Build Rose

Run build_rose.sh to install boost 1.67.0 and ROSE 0.11.145.3

First replace CHOOSE_ROSE_INSTALLATION_LOCATION with your desired installation location.

I have not been able to install ROSE if boost is not installed in the default location (/usr/local), so it may clash if you have an existing boost installation.

# Add ROSE info to path

You can either source init_rose.sh before building the graph compiler, or add it to your CLI init script (~/.bashrc, etc.)

First replace CHOOSE_ROSE_INSTALLATION_LOCATION with your desired installation location.

# Building Balor's graph compiler

Run
`make -j16` 
in the graph_compiler folder.