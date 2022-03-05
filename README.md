# Seinfeld Noisy Pixels Cleaning

I really love Seinfeld.  
I saw that Netflix uploaded the entire series and it makes me wanna watch it all over again. The main thing that has changed since the last time I have watched it is the quality of the video. I can easily notice the change in quality, the characters are much sharper and clearer. It is far better right now.
Although, there is one poor thing that has also changed - there are a lot of noisy pixels in each frame!  


I decided to fix the noisy pixels as a self challenge in order to improve my Computer Vision skills. In this paper I tend to solve this problem (remove the redundant noisy pixels) using Computer Vision techniques only, without even using deep learning methods.  
Important note: I didn't want to look much at previous works because I wanted to challenge myself, so there are probably some papers on this topic but I didn't search for them on purpose.  


Let's jump straight to business, here is a frame that demonstrates the problem pretty well:

![original_frame](https://user-images.githubusercontent.com/83128966/156884422-c67eac11-f237-4458-b0e2-bc75c861fcca.jpg)


Try looking at homogenic bright surfaces like the refrigerator near George. You will probably see tiny darkish (and brightish) dots which prevent these surfaces from being smooth with homogenic color.  
Moreover, these dots aren’t steady, they move from frame to frame. This makes the background surfaces look really noisy as the video proceeds.
Take a look at the original scene where the noise isn't steady and moves from frame to frame.

(**VIDEO**)


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
While trying to do so, I encountered the “**Bilateral filter**” which tends to solve the very same problem. This filter is also from the image blurring filters, but less popular. OpenCV explains it well: “Bilateral filtering is highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters. The Gaussian filter takes the neighborhood around the pixel and finds its Gaussian weighted average. This Gaussian filter is a function of space alone, that is, nearby pixels are considered while filtering. It doesn't consider whether pixels have almost the same intensity. It doesn't consider whether a pixel is an edge pixel or not. So it blurs the edges also, which we don't want to do.
Bilateral filtering also takes a Gaussian filter in space, but one more Gaussian filter which is a function of pixel difference. The Gaussian function of space makes sure that only nearby pixels are considered for blurring, while the Gaussian function of intensity difference makes sure that only those pixels with similar intensities to the central pixel are considered for blurring. So it preserves the edges since pixels at edges will have large intensity variation.”
(For more information about Bilateral filtering: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html)

The result using the Bilateral filtering:
![bilateral_filter](https://user-images.githubusercontent.com/83128966/156885017-b7cfd012-007b-4750-beb2-b2028b10ac36.png)


