Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA, client: :cuda)

defmodule BinaryGen do
  def generate_dataset(size, type), do: _generate_dataset(size, type, <<>>, <<>>)

  defp _generate_dataset(0, _type, a, b), do: a <> b

  defp _generate_dataset(size, type, a, b) do
    new_a = _create_bin(type)
    new_b = _create_bin(type)

    _generate_dataset(div(size,2), type, a <> new_a, b <> new_b)
  end

  defp _create_bin(type) do
    case type do
      :float_32 -> <<:rand.uniform() * 100::float-32>>
      :float_64 -> <<:rand.uniform() * 100::float-64>>
      :int_32 -> <<:rand.uniform() * 100::integer-32>>
      unknown -> raise "Provided type is invalid: #{unknown}"
    end
  end
end

defmodule DotProductNx do
  @moduledoc """
  Elixir/Nx port of the CUDA dot-product benchmark (map_2kernel + reduce_kernel).

  Structural correspondence with the CUDA source:

    CUDA                          Nx
    ----                          --
    map_2kernel(a1, a2, f)   ->   map_2kernel(a1, a2, f)   (defn, f applied elementwise)
    reduce_kernel(a, f)      ->   reduce_kernel(a, f)       (defn, f folded over a)
    anonymous_mult           ->   anonymous_mult (defn)
    anonymous_sum            ->   anonymous_sum  (defn)
    get_anonymous_*_ptr()    ->   &anonymous_*/2 (Elixir function capture, passed as "pointer")
    main()                   ->   run/1
  """

  import Nx.Defn

  # ---------------------------------------------------------------
  # "map_2kernel": elementwise binary op over two tensors, driven
  # by a function argument, exactly like the CUDA kernel takes a
  # `float (*f)(float, float)` device function pointer.
  # ---------------------------------------------------------------
  defn map_2kernel(a1, a2, f) do
    f.(a1, a2)
  end

  # ---------------------------------------------------------------
  # "reduce_kernel": fold a over f, seeded with the tensor's
  # first-element-compatible zero, mirroring the CUDA tree-reduction
  # kernel that starts from ref4[0] = 0 and repeatedly applies f.
  # ---------------------------------------------------------------
  defn reduce_kernel(a, f) do
    zero = Nx.tensor(0.0, type: Nx.type(a))
    Nx.reduce(a, zero, fn x, acc -> f.(x, acc) end)
  end

  # ---------------------------------------------------------------
  # The two "device functions" exposed as function pointers in the
  # CUDA program (anonymous_mult / anonymous_sum), here just plain
  # defn functions captured with &/2 and passed around as values.
  # ---------------------------------------------------------------
  defn anonymous_mult(a, b), do: a * b
  defn anonymous_sum(a, b), do: a + b

  @doc """
  Equivalent of CUDA's main(argc, argv): builds two random float
  arrays of size N, runs map then reduce through the function
  pointers, times the whole device-facing portion (allocation +
  transfer + kernels + transfer back, same as the CUDA cudaEvent
  timers bracket), and prints "NX\\tN\\ttime_ms".
  """
  def run(n) do
    # host arrays (CUDA: a = malloc(...); a[i] = rand();)
    #
    # Built straight into a binary (4 bytes/float, same as a C array)
    # instead of an Erlang list comprehension. A list here would cost
    # ~40 bytes/element (cons cell + boxed double) instead of 4 — a
    # ~10x blowup that OOM-kills the BEAM at large N even though the
    # equivalent CUDA host malloc is fine. :into binary comprehension
    # keeps the same per-element :rand.uniform() semantics, just with
    # a compact target instead of a list.
    # a_bin = for _ <- 1..n, into: <<>>, do: <<:rand.uniform() * 32_767::float-32-native>>
    # b_bin = for _ <- 1..n, into: <<>>, do: <<:rand.uniform() * 32_767::float-32-native>>
    a_bin = BinaryGen.generate_dataset(n, :float_32)

    IO.puts("a_bin criado: #{div(byte_size(a_bin), 4)}")

    b_bin = BinaryGen.generate_dataset(n, :float_32)

    IO.puts("b_bin criado: #{div(byte_size(b_bin), 4)}")


    f1 = &anonymous_mult/2
    f2 = &anonymous_sum/2

    start = System.monotonic_time()

    # cudaMalloc + cudaMemcpy HostToDevice equivalent: build device
    # tensors directly from the binaries, landing straight on the
    # :cuda client in one step (Nx.from_binary/3 takes a :backend
    # option, so no separate backend_transfer hop is needed). If no
    # CUDA device is visible, this raises immediately
    # (EXLA.Client.fetch!(:cuda) fails) rather than silently falling
    # back to host.
    dev_a = Nx.from_binary(a_bin, {:f, 32}, backend: {EXLA.Backend, client: :cuda})
    dev_b = Nx.from_binary(b_bin, {:f, 32}, backend: {EXLA.Backend, client: :cuda})

    # map_2kernel<<<...>>>(dev_a, dev_b, dev_resp, N, f1)
    dev_resp = map_2kernel(dev_a, dev_b, f1)

    # reduce_kernel<<<...>>>(dev_resp, d_final, f2, N)
    d_final = reduce_kernel(dev_resp, f2)

    # cudaMemcpy DeviceToHost
    final = Nx.to_number(d_final)

    stop = System.monotonic_time()
    time_ms = System.convert_time_unit(stop - start, :native, :microsecond) / 1000.0

    IO.puts("Nx\t#{n}\t#{time_ms}")

    final
  end
end

# equivalent of: int N = atoi(argv[1]);
[n_arg | _] = System.argv()
n = String.to_integer(n_arg)

DotProductNx.run(n)