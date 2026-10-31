// Read-only current settings/public camera probe. NO render, setter, or file write.
cancellationToken.ThrowIfCancellationRequested();
if ( Game.IsPlaying ) throw new InvalidOperationException( "Play must remain off." );
var project = Sandbox.Project.Current ?? throw new InvalidOperationException( "No project." );
var assets = System.IO.Path.GetFullPath( project.GetAssetsPath() ).TrimEnd( '\\', '/' );
if ( !string.Equals( assets, @"C:\Users\wattz\Documents\S&box Projects\greyfield\Assets", StringComparison.OrdinalIgnoreCase ) )
    throw new InvalidOperationException( "Expected existing Greyfield project." );
var session = Editor.SceneEditorSession.Active ?? throw new InvalidOperationException( "No active scene." );
var scene = session.Scene;
if ( scene is null || !scene.IsValid || session.HasUnsavedChanges ||
    !string.Equals( scene.Source?.ResourcePath, "modly_tests/phase22_uv_pbr/modly_phase22_uv_pbr_test.scene", StringComparison.OrdinalIgnoreCase ) )
    throw new InvalidOperationException( "Expected exact saved Phase22 scene." );
var cameras = scene.GetAllObjects( true ).Where( x => x.Name == "Phase22_Camera_top" ).ToArray();
if ( cameras.Length != 1 ) throw new InvalidOperationException( "Expected unique existing top camera." );
var camera = cameras[0].Components.Get<CameraComponent>();
if ( camera is null || !camera.IsValid ) throw new InvalidOperationException( "Camera missing." );
var settings = Editor.EditorUtility.RenderSettings;
float[] Vec( Vector3 v ) => new[] { v.x, v.y, v.z };
float[] Quat( Rotation v ) => new[] { v.x, v.y, v.z, v.w };
var transform = camera.WorldTransform;
var viewport = camera.Viewport;
var originalSize = camera.CustomSize;
var result = new
{
    Classification = "READ_ONLY_CURRENT_RENDER_SETTINGS_AND_PUBLIC_CAMERA",
    HistoricalCaptureSettingsWereNotRecorded = true,
    NoRenderOrSetterOrProjectWrite = true,
    SettingsObservation = new
    {
        AntiAliasMode = settings.AntiAliasMode.ToString(),
        AntiAliasQuality = settings.AntiAliasQuality.ToString(),
        RenderScale = settings.RenderScale,
        UpscalerMode = settings.UpscalerMode.ToString(),
        UpscalerRenderScale = settings.UpscalerRenderScale,
        Fsr3UpscalerQuality = settings.Fsr3UpscalerQuality.ToString(),
        Fsr1Sharpness = settings.Fsr1Sharpness,
        Fsr3Sharpness = settings.Fsr3Sharpness
    },
    CameraObservation = new
    {
        CameraId = camera.Id, ObjectId = cameras[0].Id,
        WorldPosition = Vec( transform.Position ), WorldRotation = Quat( transform.Rotation ), WorldScale = Vec( transform.Scale ),
        camera.Enabled, camera.EnablePostProcessing, camera.FieldOfView, FovAxis = camera.FovAxis.ToString(),
        camera.ZNear, camera.ZFar, camera.Orthographic, camera.WireframeMode,
        HasCustomProjection = camera.CustomProjectionMatrix.HasValue,
        CustomSize = originalSize.HasValue ? new[] { originalSize.Value.x, originalSize.Value.y } : null,
        Viewport = new[] { viewport.x, viewport.y, viewport.z, viewport.w },
        HasPublicAntiAliasingControl = false,
        ActualInternalSceneCameraState = "Not accessed: CameraComponent.SceneCamera is not public."
    },
    Limit = "Current settings/public saved camera only; no historical per-capture AA/native intermediate target or resolve readback."
};
if ( session != Editor.SceneEditorSession.Active || session.HasUnsavedChanges || Game.IsPlaying ||
    !Nullable.Equals( camera.CustomSize, originalSize ) )
    throw new InvalidOperationException( "Editor state changed during observation." );
return result;
