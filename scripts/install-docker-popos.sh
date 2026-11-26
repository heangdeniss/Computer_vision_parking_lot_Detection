#!/usr/bin/env bash
# Install Docker CLI + Docker Desktop UI on Pop!_OS / Ubuntu (x86_64).
# Run this from a normal host terminal (outside any Flatpak sandbox).
set -euo pipefail

if [ -f /.flatpak-info ]; then
  echo "Error: You are inside a Flatpak sandbox. Run this script from your host terminal." >&2
  exit 1
fi

ARCH="$(dpkg --print-architecture)"
if [ "$ARCH" != "amd64" ]; then
  echo "Error: Docker Desktop for Linux officially supports amd64 (found: $ARCH)." >&2
  exit 1
fi

# Pop!_OS sets ID=pop, so read UBUNTU_CODENAME (falling back to VERSION_CODENAME) for Docker's Ubuntu APT repo.
# shellcheck disable=SC1091
source /etc/os-release
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-noble}}"
echo "==> Setting up official Docker APT repository (ubuntu/${CODENAME})..."

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg pass uidmap

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

echo "==> Installing Docker CLI, Buildx, and Compose plugins..."
sudo apt-get install -y docker-ce-cli docker-buildx-plugin docker-compose-plugin

echo "==> Downloading official Docker Desktop (.deb)..."
TMP_DEB="$(mktemp --suffix=.deb)"
trap 'rm -f "$TMP_DEB"' EXIT
curl -fL "https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb" -o "$TMP_DEB"

echo "==> Installing Docker Desktop UI..."
sudo apt-get install -y "$TMP_DEB"

if getent group kvm >/dev/null 2>&1 && ! id -nG "$USER" | grep -qw kvm; then
  echo "==> Adding $USER to the kvm group (required by Docker Desktop VM)..."
  sudo usermod -aG kvm "$USER"
  echo "    Note: Log out and back in if Docker Desktop reports a /dev/kvm permission error."
fi

echo "==> Enabling and starting Docker Desktop..."
systemctl --user daemon-reload
systemctl --user enable --now docker-desktop

echo
echo "==> Installed versions:"
docker --version
docker compose version
echo
echo "Docker Desktop UI is starting (also available in your Applications menu)."
echo "Once Docker Desktop shows 'Engine running', deploy this project with:"
echo "  docker compose up -d --build"
echo "Then open: http://localhost:8000"
