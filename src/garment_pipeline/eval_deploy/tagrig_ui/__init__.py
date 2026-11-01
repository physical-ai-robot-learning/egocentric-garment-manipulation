"""Inference web UI of the tag-rig laundry policies (split out of act_tagrig_ui.py on 2026-11-01, behaviour unchanged).

Start it with `python -m garment_pipeline.eval_deploy.act_tagrig_ui [--port 8770] [--bridge-url ...] [--dry-only]`.

  settings.py   persisted UI settings (act_tagrig_ui_settings.json, same keys and defaults), checkpoint folders, heartbeat timeout
  control.py    run control: UICtl (stop / heartbeat / step tokens), App (bridge poller, watchdog, runs, motions, views), move_ready
  server.py     the Flask app (create_app: page, MJPEG views, /api/*) and main()
  page.py       page.html, served by GET /
"""
