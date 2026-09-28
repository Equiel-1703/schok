import Config

System.put_env("XLA_TARGET", System.get_env("XLA_TARGET", "cuda12"))  # was "cuda120"

config :exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: false, default_device_id: 0]