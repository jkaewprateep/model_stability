# model_stability

For study model stability, providing input in the scope of the game environment both similar and different than the training input can provide accurate results that can be trust.

1. We input noises with model prediction performance the same as radio bandwidths, ```temp = tf.random.normal([2], 0.001, 0.5, tf.float32)``` the result can be trusted and accuracy rates are proved by both scoring and game player performance.
2. Input variables we use for AI model ```temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02 ], shape=(2, 1), dtype=tf.float32))``` , the random function is to prove the equation and target possibility.

```
def random_action(  ): 
	
	gameState = p.getGameState()
	player_y_array = gameState['player_y']
	player_vel_array = gameState['player_vel']
	next_pipe_dist_to_player_array = gameState['next_pipe_dist_to_player']
	next_pipe_top_y_array = gameState['next_pipe_top_y']
	next_pipe_bottom_y_array = gameState['next_pipe_bottom_y']
	next_next_pipe_dist_to_player_array = gameState['next_next_pipe_dist_to_player']
	next_next_pipe_top_y_array = gameState['next_next_pipe_top_y']
	next_next_pipe_bottom_y_array = gameState['next_next_pipe_bottom_y']
	
	gap = (( next_pipe_bottom_y_array - next_pipe_top_y_array ) / 2 )
	top = next_pipe_top_y_array
	target = top + gap
	
	space = 512 - pipe_gap 
	upper_pipe_buttom = next_pipe_top_y_array + 0.8 * space
	
	coeff_01 = upper_pipe_buttom
	coeff_02 = 512 - player_y_array
	
	temp = tf.random.normal([2], 0.001, 0.5, tf.float32)
	# temp = tf.ones([2], tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02 ], shape=(2, 1), dtype=tf.float32))
	# temp = tf.nn.softmax(temp)
	# 
	
	temp = tf.math.argmax(temp)
	action = int(temp[0])
	
	action_name = list(actions.values())[action]
	action_name = [ x for ( x, y ) in actions.items() if y == action_name]
	
	print( "steps: " + str( step ).zfill(6) + " action: " + str(action_name) + " coeff_01: " 
          + str(int(coeff_01)).zfill(6) + " coeff_02: " 
          + str(int(coeff_02)).zfill(6) 

	)

	return action
```

## Result ##

![alt text](https://github.com/jkaewprateep/model_stability/blob/main/Figure_14.png "image Title")
![alt text](https://github.com/jkaewprateep/model_stability/blob/main/Figure_22.png "image Title")
![alt text](https://github.com/jkaewprateep/model_stability/blob/main/Figure_25.png "image Title")
![alt text](https://github.com/jkaewprateep/model_stability/blob/main/Figure_5.png "image Title")
