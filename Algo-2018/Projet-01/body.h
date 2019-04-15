#ifndef _BODY_
#define _BODY_

typedef struct _body
{
	int x;
	int y;
	float mass;
	acceleration_vector* av;
	speed_vector* sv;
}body;

/* create a new body */
body* create_body( int x, int y, float speed_x, float speed_y, float acceleration_x, float acceleration_y, float mass );

/* create the acceleration vector of b1 undergoing the strenght of b2 */
void apply_body_strength_on_body( body* b1, body* b2 );

/* free the body */
void free_body( body* b );

#endif