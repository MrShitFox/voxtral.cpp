using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

public static class VoxtralServerClient
{
    public const int ChunkSamples = 1280;
    public const int ChunkBytes = ChunkSamples * 2;

    private static void AddBearer(HttpRequestMessage request, string apiKey)
    {
        if (apiKey != "-")
        {
            request.Headers.Authorization =
                new AuthenticationHeaderValue("Bearer", apiKey);
        }
    }

    public static async Task<string> TranscribeWavAsync(
        Uri endpoint,
        string apiKey,
        string wavPath,
        CancellationToken cancellationToken)
    {
        using var http = new HttpClient();
        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint);
        AddBearer(request, apiKey);
        request.Content = new ByteArrayContent(
            await File.ReadAllBytesAsync(wavPath, cancellationToken));
        request.Content.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");

        using HttpResponseMessage response =
            await http.SendAsync(request, cancellationToken);
        string body = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException(
                $"Server returned {(int)response.StatusCode}: {body}");
        }

        using JsonDocument json = JsonDocument.Parse(body);
        return json.RootElement.GetProperty("text").GetString() ?? "";
    }

    public static async Task TranscribePcmRealtimeAsync(
        Uri endpoint,
        string apiKey,
        string pcmPath,
        CancellationToken cancellationToken)
    {
        using var socket = new ClientWebSocket();
        if (apiKey != "-")
        {
            socket.Options.SetRequestHeader("Authorization", $"Bearer {apiKey}");
        }
        await socket.ConnectAsync(endpoint, cancellationToken);

        Task receiver = ReceiveEventsAsync(socket, cancellationToken);
        byte[] configure = JsonSerializer.SerializeToUtf8Bytes(new
        {
            type = "session.configure",
            audio = new
            {
                format = "pcm_s16le",
                sample_rate = 16000,
                channels = 1,
            },
            events = new
            {
                token = false,
                partial = true,
            },
        });
        await socket.SendAsync(
            new ArraySegment<byte>(configure),
            WebSocketMessageType.Text,
            true,
            cancellationToken);

        await using (FileStream pcm = File.OpenRead(pcmPath))
        {
            var chunk = new byte[ChunkBytes];
            while (true)
            {
                int count = await pcm.ReadAsync(
                    chunk.AsMemory(0, chunk.Length), cancellationToken);
                if (count == 0)
                {
                    break;
                }
                if ((count & 1) != 0)
                {
                    throw new InvalidDataException(
                        "The PCM file has an odd byte count.");
                }
                await socket.SendAsync(
                    new ArraySegment<byte>(chunk, 0, count),
                    WebSocketMessageType.Binary,
                    true,
                    cancellationToken);

                // 1280 mono samples at 16 kHz are exactly 80 ms.
                await Task.Delay(TimeSpan.FromMilliseconds(80), cancellationToken);
            }
        }

        byte[] finish = Encoding.UTF8.GetBytes("{\"type\":\"input_audio.end\"}");
        await socket.SendAsync(
            new ArraySegment<byte>(finish),
            WebSocketMessageType.Text,
            true,
            cancellationToken);
        await receiver;
    }

    private static async Task ReceiveEventsAsync(
        ClientWebSocket socket,
        CancellationToken cancellationToken)
    {
        var receiveBuffer = new byte[16 * 1024];
        using var message = new MemoryStream();
        bool completed = false;

        while (true)
        {
            WebSocketReceiveResult result = await socket.ReceiveAsync(
                new ArraySegment<byte>(receiveBuffer), cancellationToken);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                if (socket.State == WebSocketState.CloseReceived)
                {
                    await socket.CloseOutputAsync(
                        WebSocketCloseStatus.NormalClosure,
                        "ack",
                        cancellationToken);
                }
                if (!completed)
                {
                    throw new InvalidOperationException(
                        "WebSocket closed before session.completed.");
                }
                return;
            }
            if (result.MessageType != WebSocketMessageType.Text)
            {
                throw new InvalidDataException(
                    "The server returned an unexpected binary message.");
            }

            message.Write(receiveBuffer, 0, result.Count);
            if (!result.EndOfMessage)
            {
                continue;
            }
            if (message.Length > 4 * 1024 * 1024)
            {
                throw new InvalidDataException("Server event exceeds 4 MiB.");
            }

            using JsonDocument json = JsonDocument.Parse(message.ToArray());
            message.SetLength(0);
            JsonElement root = json.RootElement;
            string type = root.GetProperty("type").GetString() ?? "";
            switch (type)
            {
                case "transcript.partial":
                    Console.Write(
                        "\r" + (root.GetProperty("text").GetString() ?? ""));
                    break;
                case "transcript.final":
                    Console.WriteLine(
                        "\r" + (root.GetProperty("text").GetString() ?? ""));
                    break;
                case "session.completed":
                    completed = true;
                    break;
                case "error":
                    throw new InvalidOperationException(
                        $"{root.GetProperty("code").GetString()}: " +
                        root.GetProperty("message").GetString());
            }
        }
    }

    public static async Task<int> Main(string[] args)
    {
        if (args.Length != 4 || (args[0] != "batch" && args[0] != "realtime"))
        {
            Console.Error.WriteLine(
                "Usage:\n" +
                "  batch URL API_KEY input.wav\n" +
                "  realtime WS_URL API_KEY input.pcm\n" +
                "Use '-' as API_KEY only for a server started with --no-auth.");
            return 2;
        }

        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            cancellation.Cancel();
        };
        try
        {
            if (args[0] == "batch")
            {
                string text = await TranscribeWavAsync(
                    new Uri(args[1]), args[2], args[3], cancellation.Token);
                Console.WriteLine(text);
            }
            else
            {
                await TranscribePcmRealtimeAsync(
                    new Uri(args[1]), args[2], args[3], cancellation.Token);
            }
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }
}
