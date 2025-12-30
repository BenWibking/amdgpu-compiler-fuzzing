FROM registry.opensuse.org/opensuse/leap:15.6

RUN zypper --non-interactive refresh && \
    zypper clean --all

COPY rocm_so.conf rocm_so.conf

RUN if ! zypper repos -E | grep devel_languages_perl; then \
    zypper --non-interactive addrepo https://download.opensuse.org/repositories/devel:/languages:/perl/15.6/devel:languages:perl.repo; \
    fi && \
    echo "[ROCm-6.4.4]" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "name=ROCm6.4.4" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "baseurl=https://repo.radeon.com/rocm/zyp/6.4.4/main" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "enabled=1" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "gpgcheck=1" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" | tee --append /etc/zypp/repos.d/rocm.repo && \
    zypper --non-interactive --gpg-auto-import-keys refresh && \
    zypper --non-interactive install --no-recommends \
        hipcc \
        hip-devel \
        rocm-device-libs \
        rocm-hip-runtime && \
    zypper clean --all && \
    cat rocm_so.conf >> /etc/ld.so.conf.d/rocm.conf && \
    rm rocm_so.conf && \
    ldconfig

ENV PATH="/opt/rocm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

# Set environment variables for ROCm
ENV ROCM_PATH=/opt/rocm
ENV hip_DIR=/opt/rocm/lib/cmake/hip
ENV AMREX_AMD_ARCH=gfx90a
ENV CC=amdclang
ENV CXX=amdclang++

# add packages for C++ builds and basic GNU tools
RUN zypper --non-interactive install --no-recommends \
    wget tar gzip file gcc gcc-c++ make cmake patch \
    which coreutils findutils grep gawk sed ripgrep \
    git python3 && \
    zypper clean --all

# create user 'leap'
RUN useradd -ms /bin/bash leap
RUN usermod -a -G render,video leap
WORKDIR /home/leap
USER leap

# Set the default command to bash
CMD [ "/bin/bash" ]
