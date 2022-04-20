The reason parallel processing makes this tricky is because we have no 
guarantees of individual pixel completion. Any synchronization needs to
be limited to retain speed.

Developing the algorithm: First I want something that will perfectly tile 
without overlapping values, for reasons stated above. Then I wanted to see 
what values I could populate with the pattern, and what values I am missing. 
From there I need to see how to share position information quickly.

Process:

V1:
    // Checking with locks
    // Stage 1:
    //  011+011
    //  001+001
    //  000 000
    //   ++/ ++  No overlaps positions written to every pixel
    //  011+011
    //  001+001
    //  000 000

    // Stage 2: write to neighbors
    //  1-- 1--
    //  10- 10-
    //  111 111
    //           No overlaps positions written to every pixel
    //  1-- 1--
    //  10- 10-
    //  111 111

    // Where there was a one an encoded position was written. 
    // Note: no overlaps

    // Stage4:
    // Each node solves connections it knows both positions for
    
    //   a b
    //   c d

    // Who solves what:
    //   -   - 
    //  a b b -
    //   ax xb 
    //            // node knows neighbor position
    //   bx xb    // solve    n center
    //  c d d -   //        n   n 
    //   c   d    //          n  

    //   #   #    // for neighbors so we need x too
    //  # b b #   // given: -     must solve: \          // line = pair
    //   ax xb    //        -                 -  
    //            //        / | |             / | \
    //   bx xb    // 
    //  # d d #   //
    //   #   #    //

V2:
    // Instead: Checking with locks:
    // Stage 1:
    //  011 011
    //  001+001
    //  001 001
    //   + x +   // No overlaps positions written to every pixel
    //  011 011  // Note: only adding center columns and 2 diagonals
    //  001+001
    //  001 001


    // Stage 2+3:  
    //   a b
    //   c d     
    //           // Each node solves:
    //  ---(b)-  //   n
    //  - b b -  //   n center
    //  -ad bb-  //   n n
    //           
    //  -ab db-  // No overlaps positions written to every pixel
    //  - d d -  // Note: only adding center columns and 2 diagonals
    //  --- ---

    // Final output:
    //  ### ###
    //  #0- -0#
    //  #|\ /|#
    //           // No overlaps positions written to every pixel
    //  #|/ \|#  // Note: only adding center columns and 2 diagonals
    //  #0- -0#
    //  ### ###

V2 will work.