#ifndef _VECTOR_
#define _VECTOR_

typedef struct _acceleration_vector
{
	float acceleration_x;
	float acceleration_y;
}acceleration_vector;

typedef struct _speed_vector
{
	float speed_x;
	float speed_y;
}speed_vector;

/* create a acceleration vector */
acceleration_vector* create_acceleration_vector( void );

/* create a speed vector */
speed_vector* create_speed_vector( void );

/* free the acceleration vector */
void free_acceleration_vector( acceleration_vector* av );

/* free the speed vector */
void free_speed_vector( speed_vector* sv );

/* add av2 into av1 */
void acceleration_vector_sum( acceleration_vector* av1, acceleration_vector* av2 );

/* add sv2 into sv1 */
void speed_vector_sum( speed_vector* sv1, speed_vector* sv2 );

/* apply the acceleration vector on the speed vector */
void apply_acceleration( acceleration_vector* av, speed_vector* sv );

/* reset acceleration vector */
void reset_acceleration_vector( acceleration_vector* sv );

/* reste speed vector */
void reset_speed_vector( speed_vector* sv );



#endif
