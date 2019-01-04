#!/bin/sh
#cropping the videos of open field of Oct_2017
#each video contains 4 animals and I want 1 vid/animal to be able to use DeepLabCut
cd /media/amr/Amr_4TB/open_field
#cut horizontally
ffmpeg -i Cuatro_*.avi -filter_complex "[0]crop=iw:ih/2:0:0[top];[0]crop=iw:ih/2:0:oh[bottom]" \
-map "[top]" top.mp4 -map "[bottom]" bottom.mp4

##################################################################################################
#the videos are not exactly similar around the midline, So I remove 10 pixels from the left
#of the top and bottom videos to make the vertical cuts even. If they are not even, there will
#be still a little bit of the left mouse apparent in the right video which might confuse the network
#bottom
ffmpeg -i bottom.mp4 -filter_complex "[0]crop=in_w-2*10:ih:ow:0[right]"  -map "[right]" Bottom.mp4

#now, cut vertically
ffmpeg -i Bottom.mp4 -filter_complex "[0]crop=iw/2:ih:0:0[left];[0]crop=iw/2:ih:ow:0[right]" \
-map "[left]" 003.mp4 -map "[right]" 004.mp4

rm bottom.mp4 Bottom.mp4
##################################################################################################
#top

ffmpeg -i top.mp4 -filter_complex "[0]crop=in_w-2*10:ih:ow:0[right]"  -map "[right]" Top.mp4



ffmpeg -i Top.mp4 -filter_complex "[0]crop=iw/2:ih:0:0[left];[0]crop=iw/2:ih:ow:0[right]" \
-map "[left]" 001.mp4 -map "[right]" 002.mp4

rm  top.mp4 Top.mp4
###################################################################################################



