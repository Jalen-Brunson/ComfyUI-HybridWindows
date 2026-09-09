# Example images

Copy these three images into `ComfyUI/input/hybrid_windows/`. The example
workflows already select those paths. The main FL2VA flow uses only the first image;
the recurring-end variant also uses the last image. Alternatively, upload the images through
the native Load Image nodes and select the uploaded filenames.

| File | Used by | Source |
|---|---|---|
| [ref2va_stock_portrait.jpg](ref2va_stock_portrait.jpg) | Ref2VA reference, shared by all three windows | [Nadine Ginzel, Pexels photo 31428197](https://www.pexels.com/photo/portrait-of-a-young-man-in-black-shirt-31428197/) |
| [fl2va_stock_first.png](fl2va_stock_first.png) | FL2VA first image, window 1 | [Kampus Production, Pexels video 8189169](https://www.pexels.com/video/man-wearing-black-long-sleeve-polo-8189169/), at 0.5 seconds |
| [fl2va_stock_last.png](fl2va_stock_last.png) | Optional FL2VA recurring ending guide, all three windows in that variant | Same clip, at 8 seconds |

The portrait is a 1536-pixel-wide JPEG downloaded from Pexels. The FL2VA frames
were extracted from the stock clip and resized to 720×1280. The full video is
not needed by either workflow. Ref2VA and FL2VA use different subjects.

These stock assets remain subject to the [Pexels License](https://www.pexels.com/license/),
separately from the repository's code license. Source and license pages were
checked on September 9, 2026. The people shown are example subjects and do not
endorse this project.
