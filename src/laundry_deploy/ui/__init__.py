"""Inference web UI of the tag-rig laundry policies on the bimanual reBot (deploy stage, Korean UI, http://localhost:8770).

Start it with `python -m laundry_deploy.ui.server [--port 8770] [--bridge-url ...] [--dry-only]`.

  settings.py      persisted UI settings (act_tagrig_ui_settings.json in TAGRIG_STATE_DIR, same keys and defaults), checkpoint
                   folders, heartbeat timeout
  run_control.py   UiRunControl: stop request / heartbeat / step tokens / latest snapshot of one run
  app.py           UiApp: bridge poller, heartbeat watchdog, runs, motions, live views and /api/status
  ready_move.py    moveReady: the checked READY move of the [READY로 이동] button
  server.py        the Flask app (page, MJPEG views, /api/*) and main()
  page.html        the page served by GET /
"""
