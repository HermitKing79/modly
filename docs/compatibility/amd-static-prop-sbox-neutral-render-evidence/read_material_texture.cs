// run_csharp async method body. REVIEW CANDIDATE ONLY: do not execute until reviewed.
// Reads exactly two already-compiled Phase 22 materials, their public shader schemas,
// and their bound compiled textures. No creation, registration, compile, Set, scene,
// camera, model, save, reflection, network, or disk-output operations.
cancellationToken.ThrowIfCancellationRequested();
if ( Game.IsPlaying ) throw new InvalidOperationException( "Probe requires Play off." );

const string projectAssetRoot = @"C:\Users\wattz\Documents\S&box Projects\greyfield\Assets";
const string relativeRoot = "modly_tests/phase22_uv_pbr/";
var specifications = new[]
{
    new { Material = "diagnostic_pointwrap_unlit.vmat", Shader = "diagnostic_pointwrap_unlit.shader", Image = "diagnostic_uv_color.png", RuntimeTexture = "g_tDiagnosticColor" },
    new { Material = "neutral_pbr.vmat", Shader = "neutral_pbr.shader", Image = "neutral_source_basecolor.png", RuntimeTexture = "g_tNeutralBaseColor" }
};
var records = new System.Collections.Generic.List<object>();

string DigestSource( string basename )
{
    var absolute = System.IO.Path.GetFullPath( System.IO.Path.Combine( projectAssetRoot, relativeRoot, basename ) );
    var allowed = System.IO.Path.GetFullPath( System.IO.Path.Combine( projectAssetRoot, relativeRoot ) ).TrimEnd( '\\', '/' ) + System.IO.Path.DirectorySeparatorChar;
    if ( !absolute.StartsWith( allowed, StringComparison.OrdinalIgnoreCase ) )
        throw new InvalidOperationException( "Source path escaped fixed Phase 22 directory." );
    var info = new System.IO.FileInfo( absolute );
    if ( !info.Exists || info.Length > 16 * 1024 * 1024 )
        throw new InvalidOperationException( "Source missing or exceeds the 16 MiB read cap: " + basename );
    using var stream = System.IO.File.OpenRead( absolute );
    return Convert.ToHexString( System.Security.Cryptography.SHA256.HashData( stream ) ).ToLowerInvariant();
}

foreach ( var spec in specifications )
{
    cancellationToken.ThrowIfCancellationRequested();
    var materialPath = relativeRoot + spec.Material;
    var asset = Editor.AssetSystem.FindByPath( materialPath );
    if ( asset is null || !asset.IsCompiledAndUpToDate || asset.IsCompileFailed )
        throw new InvalidOperationException( "Material must already be compiled and current: " + materialPath );

    var expectedAbsolute = System.IO.Path.GetFullPath( System.IO.Path.Combine( projectAssetRoot, materialPath ) );
    if ( !string.Equals( System.IO.Path.GetFullPath( asset.AbsolutePath ), expectedAbsolute, StringComparison.OrdinalIgnoreCase ) )
        throw new InvalidOperationException( "Material resolved outside its exact new asset path." );

    var material = Material.Load( materialPath );
    if ( material is null || !material.IsValid || material.IsError )
        throw new InvalidOperationException( "Invalid material: " + materialPath );
    var shader = material.Shader;
    if ( shader is null || !shader.IsValid || shader.IsError )
        throw new InvalidOperationException( "Invalid material shader: " + materialPath );
    var schema = shader.Schema;
    if ( schema is null ) throw new InvalidOperationException( "Shader schema unavailable." );

    var texture = material.GetTexture( spec.RuntimeTexture );
    if ( texture is null || !texture.IsValid || texture.IsError || !texture.IsLoaded )
        throw new InvalidOperationException( "Bound compiled texture is unavailable: " + spec.RuntimeTexture );
    var width = texture.Width;
    var height = texture.Height;
    if ( width < 1 || height < 1 || width > 1024 || height > 1024 || texture.Depth != 1 || texture.Mips < 1 )
        throw new InvalidOperationException( "Texture dimensions exceed the bounded 2D readback contract." );
    var expectedPixelCount = checked( width * height );

    var completion = new TaskCompletionSource<byte[]>( TaskCreationOptions.RunContinuationsAsynchronously );
    texture.GetPixelsAsync( pixels =>
    {
        // Color32.r/g/b/a are public byte fields, copied before callback span expires.
        try
        {
            if ( pixels.Length != expectedPixelCount )
                throw new InvalidOperationException( "Readback pixel count mismatch." );
            var bytes = new byte[checked( pixels.Length * 4 )];
            for ( int i = 0; i < pixels.Length; ++i )
            {
                var pixel = pixels[i];
                bytes[4 * i] = pixel.r;
                bytes[4 * i + 1] = pixel.g;
                bytes[4 * i + 2] = pixel.b;
                bytes[4 * i + 3] = pixel.a;
            }
            completion.TrySetResult( bytes );
        }
        catch ( Exception error )
        {
            completion.TrySetException( error );
        }
    }, 0 );

    // Public GetPixelsAsync has no cancellation argument. This is a soft wait
    // deadline, not a claim that an in-flight GPU readback can be cancelled.
    var deadline = Task.Delay( TimeSpan.FromSeconds( 8 ), cancellationToken );
    if ( await Task.WhenAny( completion.Task, deadline ) != completion.Task )
    {
        cancellationToken.ThrowIfCancellationRequested();
        throw new TimeoutException( "Eight-second compiled texture readback deadline exceeded." );
    }
    var rgba = await completion.Task;
    cancellationToken.ThrowIfCancellationRequested();
    var samples = new System.Collections.Generic.List<object>();
    for ( int row = 0; row < 5; ++row )
    {
        for ( int column = 0; column < 5; ++column )
        {
            int x = column * (width - 1) / 4;
            int y = row * (height - 1) / 4;
            int offset = 4 * (y * width + x);
            samples.Add( new { X = x, Y = y, R = rgba[offset], G = rgba[offset + 1], B = rgba[offset + 2], A = rgba[offset + 3] } );
        }
    }

    records.Add( new
    {
        MaterialPath = materialPath,
        MaterialSourceSha256 = DigestSource( spec.Material ),
        ShaderSourceSha256 = DigestSource( spec.Shader ),
        ImageSourceSha256 = DigestSource( spec.Image ),
        MaterialValid = material.IsValid,
        MaterialError = material.IsError,
        material.ShaderName,
        ShaderResourcePath = shader.ResourcePath,
        Variables = schema.Variables.Select( v => new
        {
            v.Name, v.Type, v.AttributeName, v.SourceType, v.UiType, v.UiGroup, v.UiStep,
            v.DefaultInputTexture, v.TextureEnding,
            FloatDefault = new[] { v.FloatDefault.x, v.FloatDefault.y, v.FloatDefault.z, v.FloatDefault.w },
            FloatMin = new[] { v.FloatMin.x, v.FloatMin.y, v.FloatMin.z, v.FloatMin.w },
            FloatMax = new[] { v.FloatMax.x, v.FloatMax.y, v.FloatMax.z, v.FloatMax.w },
            IntDefault = new[] { v.IntDefault.x, v.IntDefault.y, v.IntDefault.z, v.IntDefault.w }
        } ).ToArray(),
        Combos = schema.Combos.Select( c => new { c.Name, c.Group, c.Min, c.Max, c.Index, c.Values } ).ToArray(),
        RuntimeTextureName = spec.RuntimeTexture,
        TextureResourcePath = texture.ResourcePath,
        TextureValid = texture.IsValid,
        TextureError = texture.IsError,
        texture.IsLoaded,
        Width = width,
        Height = height,
        texture.Depth,
        texture.Mips,
        ImageFormat = texture.ImageFormat.ToString(),
        Flags = texture.Flags.ToString(),
        MipZeroRgbaByteCount = rgba.Length,
        MipZeroRgbaSha256 = Convert.ToHexString( System.Security.Cryptography.SHA256.HashData( rgba ) ).ToLowerInvariant(),
        Samples = samples.ToArray(),
        TextureReadbackWaitLimitSeconds = 8,
        ReadbackCancellationLimit = "Soft deadline only; underlying public API has no cancellation argument."
    } );
}

return new
{
    Classification = "READ_ONLY_PHASE22_COMPILED_SHADER_MATERIAL_TEXTURE_SNAPSHOT",
    NoSceneOrResourceMutationRequested = true,
    PixelLayout = "RGBA bytes in public GetPixelsAsync callback order; five-by-five row/column samples recorded.",
    Records = records.ToArray()
};
