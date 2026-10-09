from PIL import Image

def image_info(path):
    with Image.open(path) as im:
        d={'size':list(im.size),'mode':im.mode,'format':im.format}
        if im.mode=='RGBA':
            a=im.getchannel('A'); d['alpha_extrema']=list(a.getextrema()); d['alpha_opaque_fraction']=sum(1 for v in a.getdata() if v==255)/(im.width*im.height)
        return d

def resize_cover(im,width,height):
    im=im.convert('RGB')
    if im.size==(width,height): return im
    iw,ih=im.size; s=max(width/iw,height/ih); nw,nh=round(iw*s),round(ih*s)
    im=im.resize((nw,nh),Image.Resampling.BILINEAR)
    l=(nw-width)//2; t=(nh-height)//2
    return im.crop((l,t,l+width,t+height))


def make_transparent_white_border(source_path, output_path, border_px=64):
    """Wrap a topdown RGB image in a transparent-white RGBA border."""
    from pathlib import Path
    if border_px < 0:
        raise ValueError("border_px must be non-negative")
    source_path = Path(source_path)
    output_path = Path(output_path)
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    canvas = Image.new("RGBA", (image.width + 2 * border_px, image.height + 2 * border_px), (255, 255, 255, 0))
    canvas.alpha_composite(image.convert("RGBA"), (border_px, border_px))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "source_size": list(image.size),
        "output_size": list(canvas.size),
        "border_px": border_px,
        "border_rgb": [255, 255, 255],
        "border_alpha": 0,
        "source_alpha": 255,
        "mode": "RGBA",
    }
