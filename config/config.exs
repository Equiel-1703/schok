import Config

config :exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, default_device_id: 0]