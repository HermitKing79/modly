// Phase22C candidate: exactly one existing three-quarter camera render.
// Only temporary setters: this proof renderer's MaterialOverride and this proof
// camera's CustomSize. Root must independently review the exact file before use.
const string evidence = @"C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-static-prop-sbox-neutral-render-20261031-v1";
const string assets = @"C:\Users\wattz\Documents\S&box Projects\greyfield\Assets";
const string scenePath = "modly_tests/phase22_uv_pbr/modly_phase22_uv_pbr_test.scene";
const string modelPath = "modly_tests/phase21_static_prop/experimental_static_prop.vmdl";
const string diagnosticPath = "modly_tests/phase22_uv_pbr/diagnostic_pointwrap_unlit.vmat";
const string neutralPath = "modly_tests/phase22_uv_pbr/neutral_pbr.vmat";
const int width = 1280, height = 960;
cancellationToken.ThrowIfCancellationRequested();
if ( System.Environment.ProcessId != 7656 || Game.IsPlaying ) throw new InvalidOperationException( "Wrong process or Play state." );
var project = Sandbox.Project.Current ?? throw new InvalidOperationException( "No project." );
if ( !string.Equals( System.IO.Path.GetFullPath( project.GetAssetsPath() ).TrimEnd( '\\', '/' ), assets, StringComparison.OrdinalIgnoreCase ) )
    throw new InvalidOperationException( "Expected existing Greyfield project." );
var session = Editor.SceneEditorSession.Active ?? throw new InvalidOperationException( "No active editor scene." );
var scene = session.Scene;
if ( scene is null || !scene.IsValid || session.HasUnsavedChanges ||
    !string.Equals( scene.Source?.ResourcePath, scenePath, StringComparison.OrdinalIgnoreCase ) )
    throw new InvalidOperationException( "Expected exact saved Phase22 scene." );
if ( Editor.SceneEditorSession.All.Any( s => s.HasUnsavedChanges ) ) throw new InvalidOperationException( "All existing scenes must be saved." );

string Sha( byte[] bytes ) => Convert.ToHexString( System.Security.Cryptography.SHA256.HashData( bytes ) ).ToLowerInvariant();
byte[] ReadBounded( string path, int cap = 2 * 1024 * 1024 )
{
    var info = new System.IO.FileInfo( path );
    if ( !info.Exists || info.Length < 1 || info.Length > cap ) throw new InvalidOperationException( "Fixed input missing/oversized: " + path );
    return System.IO.File.ReadAllBytes( path );
}
float[] V( Vector3 x ) => new[] { x.x, x.y, x.z };
float[] Q( Rotation x ) => new[] { x.x, x.y, x.z, x.w };
float[] C( Color x ) => new[] { x.r, x.g, x.b, x.a };
float[] M( Matrix x ) => new[] { x.M11,x.M12,x.M13,x.M14,x.M21,x.M22,x.M23,x.M24,x.M31,x.M32,x.M33,x.M34,x.M41,x.M42,x.M43,x.M44 };
string Json( object x ) => System.Text.Json.JsonSerializer.Serialize( x, new System.Text.Json.JsonSerializerOptions { WriteIndented = true, DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never } );
var identities = new[]
{
    new { Path="modly_tests/phase21_static_prop/experimental_static_prop.vmdl", Hash="113d688135246d71cbd85c9c74dbd65af9f75160cce8efddce921c5964c8547c" },
    new { Path="modly_tests/phase21_static_prop/experimental_static_prop.vmdl_c", Hash="9a83938757967a81cb7f37d9a505ae14dc491b109ba19df42c4d2b830cfd7a11" },
    new { Path="modly_tests/phase22_uv_pbr/Modly_Phase22_UV_PBR_Test.scene", Hash="280f17fcc78c4db7c6085bb10e6f7cc2cc8b43459a303eb85168428e68890e20" },
    new { Path="modly_tests/phase22_uv_pbr/diagnostic_pointwrap_unlit.vmat", Hash="7f2571ef2f7ebafd4d077f21c9f8164b2740b994fe747fe8f05ac52e7fb46655" },
    new { Path="modly_tests/phase22_uv_pbr/diagnostic_pointwrap_unlit.vmat_c", Hash="c9ab88eb8581e96d80ba3d93c61eee1588d3ca8fc4e73997f58f488e02db89f8" },
    new { Path="modly_tests/phase22_uv_pbr/diagnostic_pointwrap_unlit.shader", Hash="b712ac40cc8445e9830cdddca141d72ee06e4d7b1c2fd258a1fd4a3033ab7982" },
    new { Path="modly_tests/phase22_uv_pbr/diagnostic_pointwrap_unlit.shader_c", Hash="ce3da90fab295ade7b6a169721f4b7befeaf5dec5b0b78de3e2a070f2d9f59d2" },
    new { Path="modly_tests/phase22_uv_pbr/diagnostic_uv_color.png", Hash="35161147faf014e2069660a8681af66762f3d3ad4cc859904087becbb328a6d3" },
    new { Path="modly_tests/phase22_uv_pbr/neutral_pbr.vmat", Hash="9f084377486c7cab460b2927127de0f2b80c4e763b0ad942a22f2a43d3b38c5b" },
    new { Path="modly_tests/phase22_uv_pbr/neutral_pbr.vmat_c", Hash="2d0b6aef6968696a667d3a41bf7af82f047dc31298195b0a5ec7f807aaaae9f7" },
    new { Path="modly_tests/phase22_uv_pbr/neutral_pbr.shader", Hash="5a79ec8b48b1f4d936da1463226c1ba2655efd5e33dfc043e718bd6b31c5fd12" },
    new { Path="modly_tests/phase22_uv_pbr/neutral_pbr.shader_c", Hash="6e958990e31caf8cd877c78ca9018262d072c2dd8774defbdca4c4f8d1aeed9a" },
    new { Path="modly_tests/phase22_uv_pbr/neutral_source_basecolor.png", Hash="2364e7ddb8b271a95db206b39bfbdf938d9e427d46dda2745357264c8451c89e" }
};
void VerifyIdentities()
{
    foreach ( var item in identities )
        if ( Sha( ReadBounded( System.IO.Path.Combine( assets, item.Path ) ) ) != item.Hash )
            throw new InvalidOperationException( "Source/compiled identity changed: " + item.Path );
}
VerifyIdentities();
var probePath = System.IO.Path.Combine( evidence, "fresh-material-texture-probe.json" );
var probeBytes = ReadBounded( probePath );
if ( Sha( probeBytes ) != "7d3016641ebf3f596895e480858d729aadb48cd8a90f3103bb97b89a995e1a93" ) throw new InvalidOperationException( "Fresh probe identity mismatch." );
using var probeDoc = System.Text.Json.JsonDocument.Parse( probeBytes );
var neutralRecord = probeDoc.RootElement.GetProperty( "Records" ).EnumerateArray().Single( x => x.GetProperty( "MaterialPath" ).GetString() == neutralPath );
if ( neutralRecord.GetProperty( "MipZeroRgbaSha256" ).GetString() != "09e5b3f39bf89b7c0a9726ed3925c75e3298e8bc5708cbc4a86646c83673d24a" ||
    !neutralRecord.GetProperty( "TextureValid" ).GetBoolean() || neutralRecord.GetProperty( "TextureError" ).GetBoolean() ||
    !neutralRecord.GetProperty( "IsLoaded" ).GetBoolean() ) throw new InvalidOperationException( "Fresh neutral texture proof failed." );
var depsPath = System.IO.Path.Combine( evidence, "mcp/01-readonly-reauthentication/stdout.txt" );
var depsBytes = ReadBounded( depsPath );
if ( Sha( depsBytes ) != "e611e41a536e3e8151fe6ef0c38bff4545831aa6e53d5cc5912552b2c7c16473" ) throw new InvalidOperationException( "Fresh dependency identity mismatch." );
var dependencyLines = System.Text.Encoding.UTF8.GetString( depsBytes ).Split( '\n' ).Select( x => x.Trim() ).Where( x => x.StartsWith( "{", StringComparison.Ordinal ) ).ToArray();
if ( dependencyLines.Length != 5 ) throw new InvalidOperationException( "Unexpected fresh dependency record format." );
using var materialInfoDoc = System.Text.Json.JsonDocument.Parse( dependencyLines[2] );
using var filesDoc = System.Text.Json.JsonDocument.Parse( dependencyLines[3] );
using var referencesDoc = System.Text.Json.JsonDocument.Parse( dependencyLines[4] );
if ( materialInfoDoc.RootElement.GetProperty( "Path" ).GetString() != neutralPath ||
    !materialInfoDoc.RootElement.GetProperty( "IsCompiledAndUpToDate" ).GetBoolean() || materialInfoDoc.RootElement.GetProperty( "IsCompileFailed" ).GetBoolean() ||
    filesDoc.RootElement.GetProperty( "UnresolvedReferences" ).GetArrayLength() != 0 || referencesDoc.RootElement.GetProperty( "Total" ).GetInt32() != 2 )
    throw new InvalidOperationException( "Fresh neutral dependency/compile prerequisites failed." );
var referencePaths = referencesDoc.RootElement.GetProperty( "Assets" ).EnumerateArray().Select( x => x.GetProperty( "Path" ).GetString() ).OrderBy( x => x ).ToArray();
if ( !referencePaths.SequenceEqual( new[] { "modly_tests/phase22_uv_pbr/neutral_pbr.shader", "modly_tests/phase22_uv_pbr/neutral_source_basecolor_png_ad7af70b.generated.vtex" }.OrderBy( x => x ) ) )
    throw new InvalidOperationException( "Fresh exact neutral references changed." );

var prop = scene.GetAllObjects( true ).Single( x => x.Name == "MODLY_PHASE22_UV_DIAGNOSTIC_ONLY" );
var cameraObject = scene.GetAllObjects( true ).Single( x => x.Name == "Phase22_Camera_three-quarter" );
var renderer = prop.Components.Get<ModelRenderer>();
var camera = cameraObject.Components.Get<CameraComponent>();
if ( prop.Id != Guid.Parse( "c0c2b231-8051-4253-936c-bfed2733b837" ) || cameraObject.Id != Guid.Parse( "2a832d21-0f09-44eb-81a3-fecd1ee5097e" ) ||
    renderer is null || camera is null || !renderer.IsValid || !camera.IsValid || !renderer.Enabled || !camera.Enabled || !prop.Enabled || !cameraObject.Enabled ||
    renderer.Id != Guid.Parse( "7bc86dd3-24f7-451b-8994-694b29881aa6" ) || camera.Id != Guid.Parse( "244bca95-5f75-4678-8712-1054e7c08d93" ) ||
    renderer.CreateAttachments || prop.Children.Any() || renderer.Model is null || renderer.Model.IsError || renderer.Model.ResourcePath != modelPath ||
    renderer.MaterialOverride is null || renderer.MaterialOverride.ResourcePath != diagnosticPath || renderer.HasMaterialGroups )
    throw new InvalidOperationException( "Exact existing proof prop/camera/diagnostic assignment or safe-attachment precondition failed." );
var originalMaterial = renderer.MaterialOverride;
var originalSize = camera.CustomSize;
var originalModel = renderer.Model;
var originalSceneObject = renderer.SceneObject;
if ( originalSceneObject is null || !originalSceneObject.RenderingEnabled || originalSceneObject.Model?.ResourcePath != modelPath )
    throw new InvalidOperationException( "Underlying existing renderer object is unavailable." );
if ( (camera.WorldTransform.Position - new Vector3( 65,-75,55 )).Length > 0.0001f || camera.FovAxis != CameraComponent.Axis.Horizontal || camera.FieldOfView != 42 ||
    camera.ZNear != 10 || camera.ZFar != 10000 || camera.Orthographic || camera.CustomProjectionMatrix.HasValue || camera.EnablePostProcessing || camera.WireframeMode ||
    camera.Viewport.x != 0 || camera.Viewport.y != 0 || camera.Viewport.z != 1 || camera.Viewport.w != 1 ) throw new InvalidOperationException( "Frozen ordinary camera settings changed." );
var neutralAsset = Editor.AssetSystem.FindByPath( neutralPath );
if ( neutralAsset is null || !neutralAsset.IsCompiledAndUpToDate || neutralAsset.IsCompileFailed ) throw new InvalidOperationException( "Neutral material is not already compiled/current." );
var neutral = Material.Load( neutralPath );
if ( neutral is null || !neutral.IsValid || neutral.ResourcePath != neutralPath ) throw new InvalidOperationException( "Neutral material did not preload validly." );
var neutralShader = neutral.Shader;
var neutralTexture = neutral.GetTexture( "g_tNeutralBaseColor" );
if ( neutralShader is null || !neutralShader.IsValid || neutralShader.ResourcePath != "modly_tests/phase22_uv_pbr/neutral_pbr.shader" ||
    neutralShader.Schema?.Variables is null || !neutralShader.Schema.Variables.Any( x => x.Name == "g_tNeutralBaseColor" ) ||
    neutralTexture is null || !neutralTexture.IsValid || neutralTexture.IsError || !neutralTexture.IsLoaded || neutralTexture.Width != 256 || neutralTexture.Height != 256 ||
    neutralTexture.Depth != 1 || neutralTexture.Mips != 9 || neutralTexture.ImageFormat != ImageFormat.RGBA8888 ) throw new InvalidOperationException( "Neutral shader/native texture prerequisite failed before application." );
var settings = Editor.EditorUtility.RenderSettings;
object GlobalPreferences() => new { AntiAliasMode=settings.AntiAliasMode.ToString(),AntiAliasQuality=settings.AntiAliasQuality.ToString(),settings.RenderScale,UpscalerMode=settings.UpscalerMode.ToString(),settings.UpscalerRenderScale };
object LightState() => new
{
    Directional = scene.GetAllComponents<DirectionalLight>().OrderBy( x => x.Id ).Select( x => new { x.Id,Name=x.GameObject.Name,x.Enabled,Position=V(x.WorldTransform.Position),Rotation=Q(x.WorldTransform.Rotation),Color=C(x.LightColor),x.Intensity,x.Temperature,x.Shadows } ).ToArray(),
    Ambient = scene.GetAllComponents<AmbientLight>().OrderBy( x => x.Id ).Select( x => new { x.Id,Name=x.GameObject.Name,x.Enabled,Color=C(x.Color) } ).ToArray()
};
object PublicState() => new
{
    Prop = new { prop.Id,prop.Enabled,Position=V(prop.WorldTransform.Position),Rotation=Q(prop.WorldTransform.Rotation),Scale=V(prop.WorldTransform.Scale),Children=prop.Children.Select(x=>x.Id).ToArray() },
    Renderer = new { renderer.Id,renderer.Enabled,Model=renderer.Model?.ResourcePath,Override=renderer.MaterialOverride?.ResourcePath,Tint=C(renderer.Tint),renderer.CreateAttachments,renderer.MaterialGroup,renderer.HasMaterialGroups,renderer.BodyGroups,renderer.LodOverride,RenderType=renderer.RenderType.ToString(),SceneObjectModel=renderer.SceneObject?.Model?.ResourcePath,SceneObjectRendering=renderer.SceneObject?.RenderingEnabled },
    Camera = new { camera.Id,camera.Enabled,ObjectEnabled=cameraObject.Enabled,Position=V(camera.WorldTransform.Position),Rotation=Q(camera.WorldTransform.Rotation),Scale=V(camera.WorldTransform.Scale),camera.FieldOfView,FovAxis=camera.FovAxis.ToString(),camera.ZNear,camera.ZFar,camera.Orthographic,camera.WireframeMode,camera.EnablePostProcessing,CustomProjection=camera.CustomProjectionMatrix.HasValue?M(camera.CustomProjectionMatrix.Value):null,CustomSize=camera.CustomSize.HasValue?new[]{camera.CustomSize.Value.x,camera.CustomSize.Value.y}:null,Viewport=new[]{camera.Viewport.x,camera.Viewport.y,camera.Viewport.z,camera.Viewport.w},Background=C(camera.BackgroundColor),ClearFlags=camera.ClearFlags.ToString() },
    Lights = LightState()
};
string Sessions() => Json( Editor.SceneEditorSession.All.OrderBy( x => x.Scene?.Source?.ResourcePath ).Select( x => new { Path=x.Scene?.Source?.ResourcePath,x.HasUnsavedChanges,Active=x==Editor.SceneEditorSession.Active } ).ToArray() );
var beforePublic = Json( PublicState() );
var beforePreferences = Json( GlobalPreferences() );
var beforeSessions = Sessions();
var pngPath = System.IO.Path.Combine( evidence, "neutral-three-quarter.png" );
var metadataPath = System.IO.Path.Combine( evidence, "neutral-render-metadata.json" );
if ( System.IO.File.Exists( pngPath ) || System.IO.File.Exists( metadataPath ) ) throw new InvalidOperationException( "Refusing smoke-output overwrite/repeat." );
byte[] png;
object captured;
var bitmap = new Bitmap( width,height );
int renderCalls = 0;
cancellationToken.ThrowIfCancellationRequested();
try
{
    renderer.MaterialOverride = neutral;
    camera.CustomSize = new Vector2( width,height );
    if ( !ReferenceEquals( renderer.MaterialOverride,neutral ) || renderer.HasMaterialGroups || !ReferenceEquals( renderer.SceneObject,originalSceneObject ) ||
        !ReferenceEquals( renderer.Model,originalModel ) || session.HasUnsavedChanges || Game.IsPlaying ) throw new InvalidOperationException( "Temporary neutral application did not settle safely." );
    var appliedTexture = renderer.MaterialOverride.GetTexture( "g_tNeutralBaseColor" );
    if ( appliedTexture is null || appliedTexture.IsError || !appliedTexture.IsValid || appliedTexture.Index != neutralTexture.Index ) throw new InvalidOperationException( "Actual temporary neutral binding differs." );
    cancellationToken.ThrowIfCancellationRequested();
    ++renderCalls;
    camera.RenderToBitmap( bitmap,false );
    // Resolve public camera metadata immediately after this sole ordinary render.
    var projection = camera.ProjectionMatrix;
    var rect = camera.ScreenRect;
    var view = camera.View;
    var frameTexture = renderer.MaterialOverride.GetTexture( "g_tNeutralBaseColor" );
    if ( !ReferenceEquals( renderer.MaterialOverride,neutral ) || !ReferenceEquals( renderer.SceneObject,originalSceneObject ) || frameTexture is null ||
        frameTexture.IsError || !frameTexture.IsValid || frameTexture.Index != neutralTexture.Index || session.HasUnsavedChanges ) throw new InvalidOperationException( "Capture-time neutral renderer/binding state changed." );
    captured = new
    {
        State = PublicState(),ProjectionMatrix=M(projection),ScreenRect=new[]{rect.Left,rect.Top,rect.Width,rect.Height},
        View = new { Position=V(view.Position),Rotation=Q(view.Rotation),view.FieldOfView,view.ZNear,view.ZFar },
        NeutralMaterialResource=neutral.ResourcePath,NeutralShaderResource=neutralShader.ResourcePath,RuntimeTextureResourcePath=frameTexture.ResourcePath,
        RuntimeTextureIndex=frameTexture.Index,frameTexture.Width,frameTexture.Height,frameTexture.Depth,frameTexture.Mips,ImageFormat=frameTexture.ImageFormat.ToString(),TextureFlags=frameTexture.Flags.ToString(),
        ManagedOverrideReferenceEqualsPreloadedNeutral=true,
        NativeSyncEvidence="Installed public setter synchronously calls UpdateObject then SceneObject.SetMaterialOverride then native SetMaterialOverrideForMeshInstances before RenderToBitmap. No independent public native global-material getter is exposed/used.",
        CurrentGlobalPreferences=GlobalPreferences(),GlobalPreferenceLimit="Current user preferences only; no effective per-view MSAA/CMAA2/resolve or historical bitmap state measured. No global setter called."
    };
    png = bitmap.ToPng();
}
finally
{
    try { renderer.MaterialOverride = originalMaterial; }
    finally { camera.CustomSize = originalSize; }
}
if ( renderCalls != 1 || !ReferenceEquals( renderer.MaterialOverride,originalMaterial ) || !Nullable.Equals( camera.CustomSize,originalSize ) ||
    !ReferenceEquals( renderer.Model,originalModel ) || !ReferenceEquals( renderer.SceneObject,originalSceneObject ) ||
    beforePublic != Json(PublicState()) || beforePreferences != Json(GlobalPreferences()) || beforeSessions != Sessions() ||
    session != Editor.SceneEditorSession.Active || session.HasUnsavedChanges || Game.IsPlaying || System.Environment.ProcessId != 7656 )
    throw new InvalidOperationException( "Exact original public/reference/saved/settings state was not restored; no image output allowed." );
VerifyIdentities();
cancellationToken.ThrowIfCancellationRequested();
var metadata = new
{
    Classification="PHASE22C_ONE_NEUTRAL_PBR_RENDER_SMOKE",Parent="9a7efb13495fc9b44d02bbbfe4fb94c5e1534839",RenderCalls=renderCalls,Width=width,Height=height,
    CameraName="Phase22_Camera_three-quarter",ImagePath=pngPath,ImageSha256=Sha(png),Capture=captured,
    FreshNativeTextureProbe=new{Path=probePath,Sha256=Sha(probeBytes),MipZeroRgbaSha256="09e5b3f39bf89b7c0a9726ed3925c75e3298e8bc5708cbc4a86646c83673d24a"},
    FreshDependencies=new{Path=depsPath,Sha256=Sha(depsBytes),RuntimeTexturePathMayBeNull=true},
    OriginalPublicState=System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(beforePublic),
    OriginalCurrentPreferences=System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(beforePreferences),
    ExactOriginalMaterialReferenceRestored=true,ExactNullableCustomSizeRestored=true,AllCapturedOriginalPublicStateRestored=true,AllSceneSavedFlagsRestored=true,SourceCompiledSceneHashesUnchanged=true,
    NoSceneOrAssetSave=true,NoSourceCompileOrGeometryChange=true,NoGlobalSettingChange=true,
    Qualification="One ordinary neutral rendering smoke image only. No BRDF/roughness appearance equivalence, intermediate sRGB measurement, all-mip fidelity, UV requalification, production readiness or user appearance approval. Phase22 NO-GO and Phase11 NO-GO remain unchanged."
};
using ( var file = new System.IO.FileStream(pngPath,System.IO.FileMode.CreateNew,System.IO.FileAccess.Write,System.IO.FileShare.None) ) file.Write(png);
using ( var file = new System.IO.FileStream(metadataPath,System.IO.FileMode.CreateNew,System.IO.FileAccess.Write,System.IO.FileShare.None) )
using ( var text = new System.IO.StreamWriter(file,new System.Text.UTF8Encoding(false)) ) text.Write(Json(metadata));
return bitmap;
