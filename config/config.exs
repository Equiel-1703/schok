import Config

config :exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: false, default_device_id: 0]