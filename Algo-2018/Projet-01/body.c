#include <stdlib.h>
#include <math.h>
#include "body.h"

#define G 0.000000000066740831

body* create_body( int x, int y, float speed_x, float speed_y, float acceleration_x, float acceleration_y, float mass )
{
	body* new_body = malloc(sizeof(body));
	if ( body == NULL )return NULL;

	new_body->x = x;
	new_body->y = y;
	new_body->mass = mass;

	new_body->av = create_acceleration_vector();
	if ( new_body->av == NULL )
	{
		free(new_body);
		return NULL;
	}
	else
	{
		new_body->av->acceleration_x = acceleration_x;
		new_body->av->acceleration_y = acceleration_y;
	}

	new_body->sv = create_speed_vector();
	if ( new_body->sv == NULL )
	{
		free(new_body);
		return NULL;
	}
	else
	{
		new_body->av->speed_x = speed_x;
		new_body->av->speed_y = speed_y;
	}

	return new_body;
}

void apply_body_strength_on_body( body* b1, body* b2 )
{
  double strength;
  double d;
  d = sqrt(pow( (b2->x - b1->x), 2 ) - pow( (b2->y - b1->y), 2 ));
  strength = G*((b1->mass * b2->mass)/pow(d,2));
  
}

void free_body( body* b )
{
	if ( b != NULL )
	{
		if ( b->av != NULL ) free(b->av);
		if ( b->sv != NULL ) free(b->sv);

		free(b);
	}
}
