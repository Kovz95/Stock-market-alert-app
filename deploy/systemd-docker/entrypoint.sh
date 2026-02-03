#!/bin/bash
# Entrypoint for systemd-based container: enable units and run systemd as PID 1.
# Services (stockalert-app, stockalert-scheduler, etc.) run as appuser via unit User=.
set -e
for unit in stockalert-app stockalert-scheduler stockalert-hourly stockalert-futures stockalert-watchdog; do
    [ -f "/etc/systemd/system/${unit}.service" ] && systemctl enable "${unit}.service" || true
done
exec /lib/systemd/systemd --system --unit=multi-user.target
