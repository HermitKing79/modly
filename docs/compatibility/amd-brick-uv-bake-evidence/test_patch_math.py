import unittest
import numpy as np
from patch_math import chart, rasterize, bilinear, srgb_decode, srgb_encode, source_padding, assert_injective


class ChartTests(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[0,0,0],[0,1,0],[0,1,1],[0,0,1]],np.float32)
        self.f=np.array([[0,1,2],[0,2,3]],np.int32)

    def test_complete_square_cross_face_centers(self):
        uv=chart(self.v,self.f,32,2); ids,b=rasterize(uv,self.f,32)
        self.assertEqual(int((ids>=0).sum()),28*28)
        np.testing.assert_allclose(b[ids>=0].sum(1),1,atol=1e-7)
        xy=np.argwhere(ids>=0)
        for y,x in xy[::73]:
            point=b[y,x]@uv[self.f[ids[y,x]]]
            np.testing.assert_allclose(point,(np.array([x,y])+.5)/32,atol=1e-7)

    def test_overlap_rejected(self):
        uv=chart(self.v,self.f,32,2)
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            rasterize(uv,np.vstack((self.f,self.f)),32)

    def test_wrong_winding_and_bad_bounds(self):
        reversed_faces=self.f[:,::-1].copy()
        uv=chart(self.v,reversed_faces)
        self.assertEqual(assert_injective(uv,reversed_faces)['positive_area_overlap_count'],0)
        mixed=self.f.copy();mixed[1]=mixed[1,::-1]
        with self.assertRaises(ValueError):chart(self.v,mixed)
        with self.assertRaises(ValueError):chart(self.v,np.array([[0,1,9]],np.int32))

    def test_linear_image_and_v_orientation(self):
        y,x=np.mgrid[:8,:8]; a=np.stack((x+.5,y+.5),-1).astype(float)/8
        uv=np.array([[.25,.75],[.61,.37]])
        np.testing.assert_allclose(bilinear(a,uv),uv,atol=1e-12)

    def test_analytic_shared_edge_contact_allowed(self):
        self.assertEqual(assert_injective(chart(self.v,self.f),self.f)['positive_area_overlap_count'],0)

    def test_subtexel_overlap_rejected_analytically(self):
        # No texel center samples these triangles, but their interiors overlap.
        uv=np.array([[.1000,.1000],[.1003,.1000],[.1000,.1003],
                     [.1001,.1001],[.1004,.1001],[.1001,.1004]],np.float32)
        f=np.array([[0,1,2],[3,4,5]],np.int32)
        ids,_=rasterize(uv,f,32);self.assertTrue((ids<0).all())
        with self.assertRaisesRegex(ValueError,'Analytic'):assert_injective(uv,f)

    def test_actual_padding_and_png_encoding_preserve_learned_covered_pixels(self):
        from sampler import pack_native_textures
        from PIL import Image
        import io
        v=np.zeros((16,16,6),np.float32);mask=np.zeros((16,16),bool);mask[4:12,4:12]=True
        v[mask]=[.71,.31,.12,-.002,1.002,.998]
        base,mr,oldbase,oldmr=source_padding(v,mask)
        expected=pack_native_textures(v)
        np.testing.assert_array_equal(base[mask],expected['base_color_rgba'][mask])
        np.testing.assert_array_equal(mr[mask],expected['metallic_roughness_rgb'][mask])
        self.assertTrue((base[mask,3]==254).all())
        stream=io.BytesIO();Image.fromarray(base[::-1]).save(stream,format='PNG');stream.seek(0)
        np.testing.assert_array_equal(np.asarray(Image.open(stream))[::-1],base)

    def test_srgb_linear_light_filter_is_not_encoded_filter(self):
        image=np.array([[[.1,.1,.1],[.9,.9,.9]]])
        uv=np.array([[.5,.5]])
        value=srgb_encode(bilinear(srgb_decode(image),uv))
        self.assertGreater(float(value[0,0]),.6)
        np.testing.assert_allclose(srgb_encode(srgb_decode(image)),image,atol=1e-12)


if __name__=='__main__':unittest.main()
