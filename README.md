# Seinfeld Noisy Pixels Cleaning

I really love Seinfeld.  
I saw that Netflix uploaded the entire series and it makes me wanna watch it all over again. The main thing that has changed since the last time I have watched it is the quality of the video. I can easily notice the change in quality, the characters are much sharper and clearer. It is far better right now.
Although, there is one poor thing that has also changed - there are a lot of noisy pixels in each frame!  


I decided to fix the noisy pixels as a self challenge in order to improve my Computer Vision skills. In this paper I tend to solve this problem (remove the redundant noisy pixels) using **Computer Vision techniques only**, without even using deep learning methods.  
Important note: I didn't want to look much at previous works because I wanted to challenge myself, so there are probably some papers on this topic but I didn't search for them on purpose.  


Let's jump straight to business, here is a frame that demonstrates the problem pretty well:

![original_frame](https://user-images.githubusercontent.com/83128966/156884422-c67eac11-f237-4458-b0e2-bc75c861fcca.jpg)


Try looking at homogenic bright surfaces like the refrigerator near George. You will probably see tiny darkish (and brightish) dots which prevent these surfaces from being smooth with homogenic color.  
Moreover, these dots aren’t steady, they move from frame to frame. This makes the background surfaces look really noisy as the video proceeds.
Take a look at the original scene where the noise isn't steady and moves from frame to frame.


https://user-images.githubusercontent.com/83128966/156885985-4d9d521f-eded-4543-add8-82a8f1d49c7e.mp4


LETS SOLVE IT !  


## Problem Definition
In this paper I work with videos (only the frames of the videos, without the sound, to be more exact). The output should be a noiseless video (while keeping the high quality resolution of the video) - a video without noisy pixels which harm the quality of the video.
The main challenge - use only Computer Vision techniques (without using Machine Learning/Deep Learning methods) in order to solve this problem.
This is a kind of “unsupervised problem”, not in the meaning of learning, but in the meaning of data without the solution in hand.


## Getting the Data
I tried looking on the internet for a Seinfeld episode which demonstrates the problem the best. I wanted to use the original Netflix’ content but unfortunately I couldn’t reach it.
Fun fact - I tried downloading the episode from Netflix (there’s a download option in the Desktop app), but unsurprisingly the downloaded content is encrypted and can only be decrypted by Netflix.
So I searched and searched and finally found a video with resolution of 1080x1920, and 24 FPS. I must say that the video demonstrates the problem pretty well, but the original content on Netflix is much noisier. 


## The Research

### First Part

I’m always trying to be practical: trying the easy and fast solutions first, examining the results, trying more techniques if I’m not satisfied by the results and so on.
Let’s examine the research process, starting from the easy solutions.

I started by trying to define the noise, searching for some kind of pattern. It was really hard, so I assumed (in order to start with something) that the noise is some kind of “**salt and pepper**” noise (https://en.wikipedia.org/wiki/Salt-and-pepper_noise). I obviously tried the “**median filter**”. This filter takes the median value of a kernel (“window of pixels”). Trying this filter I hope to remove the noise while keeping the original values of the area of the noise. I used a kernel size of 5 and got the following result (the image on top is the original frame, and the bottom image is the filtered one):

![median_filter](https://user-images.githubusercontent.com/83128966/156884865-c00d12d2-445a-4956-b59c-8d950e0a80e8.png)


I recognized 2 interesting effects: there isn’t much noise, but the frame is too blurry (you can see the blurriness on George’s face and the “**Utility**” caption on the refrigerator). I chose a kernel size of 5 because I tried a kernel size of 3 which was less blurry but more noisy.
This is the drawback of the median filter, it blurs contours and edges (like the “Utility” caption). Moreover, the video is still a bit “jumpy” when we play the video frame by frame (there’s less noise but backgrounds still change colors from frame to frame).

I wanted to try an “**average filter**” because it was worth a shot. I thought that averaging the area of a noisy pixel would blur it well while blurring contours and edges less than the median filter effect. The result was pretty much like the median filter.
I must mention the popular “**gaussian filter**”. I thought of using it but it didn’t make much sense because it’s practically a weighted average filter which gives more weight to the centralized pixels. It’s a bad idea because I want to remove the noise and not give it more weight.

I used the popular filters for image blurring, which are really easy to implement (using OpenCV), but now it’s time to move forward. 
The best shot that we got so far was the median filter, which clear the noise pretty well but blurs contours and edges. I wanted to solve this specific problem. I tried thinking of implementing my own filter which gives more weight to edges and contours, while preserving the effect of median/average filters on noisy pixels. I thought of calculating the approximated first derivative of the image, and defining contours with some thresholds. Then weigh the edgy pixels more than others.  

While trying to do so, I encountered the “**Bilateral filter**” which tends to solve the very same problem. This filter is also from the image blurring filters, but less popular. OpenCV explains it well:  
_“Bilateral filtering is highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters. The Gaussian filter takes the neighborhood around the pixel and finds its Gaussian weighted average. This Gaussian filter is a function of space alone, that is, nearby pixels are considered while filtering. It doesn't consider whether pixels have almost the same intensity. It doesn't consider whether a pixel is an edge pixel or not. So it blurs the edges also, which we don't want to do.  
Bilateral filtering also takes a Gaussian filter in space, but one more Gaussian filter which is a function of pixel difference. The Gaussian function of space makes sure that only nearby pixels are considered for blurring, while the Gaussian function of intensity difference makes sure that only those pixels with similar intensities to the central pixel are considered for blurring. So it preserves the edges since pixels at edges will have large intensity variation.”_  
(For more information about Bilateral filtering: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html)

The result using the Bilateral filtering:
![bilateral_filter](https://user-images.githubusercontent.com/83128966/156885017-b7cfd012-007b-4750-beb2-b2028b10ac36.png)


Well, it’s pretty cool !  
We got a noiseless image with smooth edges, but it’s kind of cartoonish. Some of the bars on the refrigerator look like it was painted with a brush and George’s face has totally smoothed. It happened because I chose large values for `sigmaColor` and `sigmaSpace` (see OpenCV documentation: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed). A larger value of  `sigmaColor` means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color (mainly causing the cartoonish look). Moreover, a larger value of `sigmaSpace` means that farther pixels will influence each other as long as their colors are close enough.  
I tried to use a lower value of `sigmaColor`, but I got more noise than before but less smoothing. I had another idea of filtering the image **twice** with a lower value of `sigmaColor`, this way we handle the noise in 2 batches.  
I got the result:

![twice_bilateral_with_5_13](https://user-images.githubusercontent.com/83128966/156885390-e879d3e6-104e-4c4e-8348-6c5a03e097ba.png)


This approach gave us a bit more noise than before but some fine smoothing!  
Look at George’s face once again, there is much more texture than before (at the cost of some noise).

Let’s compare the results so far. On the following figure we can see the median filter on top, the bilateral filter with large values in the middle and the bilateral filter with lower values and twice filtering on the bottom:

![combined_median_bilateral_and_twice_bilateral](https://user-images.githubusercontent.com/83128966/156885548-6bb8830a-deff-4018-a0c2-3314402545a1.png)


In my opinion, the last method of using 2 bilateral filters was the best so far. 
Though, it’s still a little bit cartoonish and we used the values of the pixels to smooth the image. This caused the image to change its original values.
Is there another way of fixing the image without using filters that change the original values of the pixels?
This was the first part of the research, the part where I tried some popular approaches of noise reduction and smoothing.


#### Second Part

In the second part of the research I’m going to try my own approach of solving the problem, using the fact that I’m working on **video** and can use **consecutive frames** to fix a specific frame.

My hypothesis was that noise changes from frame to frame and it isn't steady, means that if a given frame (f<sub>i</sub>) as a noisy pixel, then the same pixel most probably isn't noisy in the previous (f<sub>i-1</sub>) and next (f<sub>i+1</sub>) frames.  
The most simple solution which examines this hypothesis is to calculate the **median** between _x_ previous and next consecutive frames ([f<sub>i-x</sub>, ..., f<sub>i-1</sub>, f<sub>i</sub>, f<sub>i+1</sub>, ..., f<sub>i+x</sub>]).

I tried _x=2_ and got the result:  

![2_prev_2_next_median](https://user-images.githubusercontent.com/83128966/156886679-dc98e322-599c-42dc-9959-22c66247dee4.png)

We get really good results for backgrounds, but a bit of poor results for moving objects. That's not so surprising because static backgrounds have the same pixels values between frames while moving objects change their pixels values from frame to frame.  
Another drawback is that we can't fix the first and last _x_ frames because we have no reference for them.  
A possible solution is to take as a reference only the next _x_ frames while working on the first _x_ frames, and do the opposite for the last _x_ frames.  
Note that the cleaned pixels have their original value! This is a change considering the best result we got so far od using Bilatearl filtering (which changes the original pixels values).

I wanted to try using average filtering with a small kernel on consecutive frames because I thought that it might blur the moving objects just fine, while keeping the smoothness of the static objects.  
In the next figure we can see the average filtering result on top and median filtering on the bottom:

![prev_next_top_average_bottom_median](https://user-images.githubusercontent.com/83128966/156888368-6b940d8e-ea8b-402d-8e12-5794c7833a28.png)

We can see the duplication of the dynamic objects in the average filtered image (you can see the duplicate reflection of the light on George’s glasses), which is happening because of too low FPS rate. On the other hand, on the median filtered image, some of the original pixels have “disappeared” (look at the same reflection of the light).

So how can we improve the given result?  
I thought that we got pretty good results for static objects (like backgrounds) and I would like to keep it like this.  
So now I need to determine the moving objects and fix them.
There are some possible options:
1. Thresholding the difference between the median filtered frame and the original frame
2. Thresholding the difference between the average filtered frame and the original frame
3. Use SIFT algorithm in order to match features between dynamic ojects in each frame

I will start with the first option.  
Now I'm working on the half-fixed frames, frames which are already processed by the median filter on consecutive frames.  
I wanted to find the pixels that have changed from the processed frame to the original frame. In order to do that I measured the absolute difference between the processed frame to the original frame (for each pixel). Then I chose the best threshold which almostly keeps the static pixels unchained and points out the dynamic ones.  
Take a look at the masked frame:

![median_mask](https://user-images.githubusercontent.com/83128966/156889279-9e5a3d0c-93ae-4c77-8808-9344f9d17de9.png)

Now, we can use the masked pixels and take the same pixels from the original frame. I'm taking the original pixels that might be noisy because I think that there aren't much pixels that one can notice the difference.  
The resulting frame:  

![prev_next_median](https://user-images.githubusercontent.com/83128966/156891249-dc99ea3c-f6a8-4de5-a8c0-83a907e1abb8.png)

We get nice frame but it's a little buggy, look at the frame (no pan intended :) ) of George's glasses which got thiner. George's forehead is still too blurry.  
In order to fix this I wanted to take more of the dynamic pixels. One way of doing so is to increase the threshold, but it causing the background to be more nosisy. Another wat is to take the pixels that surround the masked pixels that we found above.  

I chose to find a better way of defining dynamic dynamic pixels.  
In order to define it better I had to explore the influence of the median and average filters in the dynamic objects.  
Take a look at the following figure:  

![Seinfeld Ilustration](https://user-images.githubusercontent.com/83128966/156891620-0cda7516-80e3-4c15-9fc5-c797cb3d2a53.png)

On the upper part we can see the situation where we work on the “current frame” (top center), with its previous frame (top left) and next frame (top right). The blue line demonstrates the pixels that have been moved from frame to frame (it moves downwards while the frames go by).  
On the bottom of the figure we can see the fixed frame after it has been processed with each filter (left - median, right - average).  
I noticed that the median filter disables all the moving pixels, while the average filter keeps all the pixels from all frames.  
Recap that we compare the fixed frame to the original frame, therefore the moving pixels masks will be almost like the bottom part of the figure. It means that the average filter will detect dynamic pixels better than the median filter.

The next step is to compare the masks of both filters. On the top of the next frame we can see the average filter mask and on the bottom the median filter mask.  
We can see that the average filter is much better because it detects much less static pixels and much more dynamic pixels. Moreover, the dynamic pixels are thick and grabbed together, showing the movements really good.  

![average_vs_median_mask](https://user-images.githubusercontent.com/83128966/156892285-481a8d77-28b6-43f2-9bfa-c7b8a85a6831.png)

In order to get the best result we will filter the video using the median filter on consecutive frames and take the dynamic pixels of the original frame using the avergae filter mask. The result we got is:  

![median_plus_average](https://user-images.githubusercontent.com/83128966/156892356-5eaa9938-9750-436f-8a83-f238de419b56.png)  

We got great results !  
We can see that George's forehead is not as noisy as before and not too blurred, the frame of the glasses is almost as in the original frame and the face in total is just a bit buggy. 


## Results
The best methods we got so far are the "Bilateral filtering" and the "Consecutive median filter with average mask".

Now it's time to look at some full videos.  
First, I will show the result of Bilateral filtering. Interesting note - I tried the "multiple Bilateral filtering" that I showed before and the result wasn't so good. That is because the filter performs not so great on video (though it looks fine per frame).  
So instead, I'm showing the regular Bilateral filter which make the frames look cartoonish:


https://user-images.githubusercontent.com/83128966/157315613-abb8929c-931d-4725-bce1-fbc8801f0555.mp4



https://user-images.githubusercontent.com/83128966/157313863-f64529ad-6ce8-4a8d-8172-af338b571a76.mp4



The second filter in the "Consecutive median filter with average mask":

https://user-images.githubusercontent.com/83128966/156898184-980407f8-372b-4f92-91d1-5df993e614ef.mp4

I really love that the background is steady and has its original values.  
And now for the full resolution video:

https://user-images.githubusercontent.com/83128966/156898253-009492c5-0395-4ac9-addc-5e7ba172f7f2.mp4


## Conclusions

Well, I'm pretty satisfied by the result !  
I know that the final result isn't perfect and requires more work but for the sake of the challenge it is quite good.  

We saw 2 different approaches, one used popular and easy-to-use filters (median, average, bilateral filter) and the other one was more sophisticated, which used the utility of other frames that can help processing another frame. All the solutions have their own trade-offs, some are easy-to-use and fast, others are more accurate but sophisticated and slow.  

There is much future work left to do in order to make it perfect :)

Feel free to contribute to the project.
You are more than welcome to contact me on LinkedIn (https://www.linkedin.com/in/yuval-maymon-007a36173/) and ask questions.
Thanks for reading and don't forget to give this project a star if you liked it !
