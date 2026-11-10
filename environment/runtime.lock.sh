# Pinned workflow runtime for spatial-omics-toolkit: OpenJDK 21, Nextflow 24.10.6, MCMICRO revision.
# Source this after exporting SPATIAL_TOOLKIT_SOFTWARE_ROOT, the install root that holds envs/, bin/ and libexec/.
# This file only sets pins and PATH. It installs nothing; see environment/README.md for what is not covered.
export JAVA_HOME="$SPATIAL_TOOLKIT_SOFTWARE_ROOT/envs/java-21.0.12.1+1"
export NXF_JAVA_HOME="$JAVA_HOME"
export NXF_VER=24.10.6
export MCMICRO_REVISION=92c7e027dab8972d26861286a6d1b1eb8de3bbaa
export JAVA_ARCHIVE=OpenJDK21U-jdk_x64_linux_hotspot_21.0.12.1_1.tar.gz
export JAVA_ARCHIVE_URL='https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_linux_hotspot_21.0.12.1_1.tar.gz'
export JAVA_ARCHIVE_SHA256=ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94
export PATH="$SPATIAL_TOOLKIT_SOFTWARE_ROOT/bin:$SPATIAL_TOOLKIT_SOFTWARE_ROOT/libexec/apptainer:$JAVA_HOME/bin:/opt/apptainer/current/bin:$PATH"
