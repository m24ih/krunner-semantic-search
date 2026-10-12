Name:           krunner-semantic-search
Version:        0.1.0
Release:        1%{?dist}
Summary:        Fast, local, privacy-first semantic file search plugin for KDE Plasma 6 KRunner

License:        MIT
URL:            https://github.com/m24ih/krunner-semantic-search
Source0:        %{url}/archive/v%{version}/%{name}-%{version}.tar.gz

BuildArch:      noarch

BuildRequires:  python3-devel
BuildRequires:  python3-pip
BuildRequires:  python3-build
BuildRequires:  python3-installer
BuildRequires:  python3-wheel
BuildRequires:  python3-hatchling
BuildRequires:  systemd-rpm-macros
BuildRequires:  desktop-file-utils

Requires:       python3
Requires:       python3-pyqt6
Requires:       python3-pillow
Requires:       plasma-workspace
Requires:       kf6-krunner
Requires:       python3-dbus-next
Recommends:     python3-llama-cpp-python
%{?systemd_requires}

%description
KRunner Semantic Search is a 100%% local, privacy-first semantic file search
plugin for KDE Plasma 6 KRunner powered by EmbeddingGemma 2.
It provides natural language file and document search directly from KRunner
using vector embeddings and SQLite vector search.

%prep
%autosetup -p1

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files krunner_semantic

# Install KRunner D-Bus runner plugin descriptor
install -D -p -m 0644 desktop/plasma-runner-semantic.desktop \
    %{buildroot}%{_datadir}/krunner/dbusplugins/plasma-runner-semantic.desktop

# Install application launcher for settings dialog
install -D -p -m 0644 desktop/krunner-semantic-config.desktop \
    %{buildroot}%{_datadir}/applications/krunner-semantic-config.desktop
sed -i 's|^Exec=.*|Exec=%{_bindir}/krunner-semantic-config|' \
    %{buildroot}%{_datadir}/applications/krunner-semantic-config.desktop

# Install systemd user service
install -D -p -m 0644 systemd/user/krunner-semantic.service \
    %{buildroot}%{_userunitdir}/krunner-semantic.service
sed -i 's|^ExecStart=.*|ExecStart=%{_bindir}/krunner-semantic-daemon|' \
    %{buildroot}%{_userunitdir}/krunner-semantic.service

%check
desktop-file-validate %{buildroot}%{_datadir}/applications/krunner-semantic-config.desktop || true

%post
%systemd_user_post krunner-semantic.service

%preun
%systemd_user_preun krunner-semantic.service

%postun
%systemd_user_postun_with_restart krunner-semantic.service

%files -f %{pyproject_files}
%license LICENSE
%doc README.md
%{_bindir}/krunner-semantic-daemon
%{_bindir}/krunner-semantic-indexer
%{_bindir}/krunner-semantic-config
%{_datadir}/krunner/dbusplugins/plasma-runner-semantic.desktop
%{_datadir}/applications/krunner-semantic-config.desktop
%{_userunitdir}/krunner-semantic.service

%changelog
* Mon Oct 12 2026 Melih AK <m24ih> - 0.1.0-1
- Initial Fedora RPM release for KRunner Semantic Search
