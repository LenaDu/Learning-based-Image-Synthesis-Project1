## Usage

The input directory stores 3-channel gray images. The output directory will be created if there is no such directory yet.

``` shell
python main.py --indir inputdir --outdir outdir
```

The default output data text file name is *output_data.txt*, the text file name can be indicated directly by running:

```shell
python main.py --indir inputdir --outdir outdir --textname shift_data
```



## Implementation

#### Score Image Matching

- SSD

  ```python
  ssd = np.sum(np.square(image1 - image2))
  ```

- **NCC (Better performance)**

  ```python
  ncc = np.sum((image1 / np.linalg.norm(image1)) * (image2 / np.linalg.norm(image2)))
  ```



#### Feature 

* Whole image (+SSD)
* Whole image (+NCC)
* **Edge detection (+NCC)**



#### Auto-cropping

```python
cropped_image = auto_cropping(img, rescale_factor)
```



#### Auto-white-balancing

```python
balanced_image = auto_white_balance(img)
```



## Appendix

Homework page: https://learning-image-synthesis.github.io/sp22/assignments/hw1



