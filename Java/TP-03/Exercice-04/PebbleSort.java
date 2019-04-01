import java.util.Objects;
import java.util.NoSuchElementException​;

public class PebbleSort
{
	public static void swap(int[] array,int index1,int index2)
	{
		int buff;
		buff = array[index1];
		array[index1] = array[index2];
		array[index2] = buff;
	}

	public static int indexOfMin(int[] array)
	{
		Objects.requireNonNull(array);
		return PebbleSort.indexOfMin(array,0,array.length);
	}

	public static int indexOfMin(int[] array, int index1, int index2 )
	{
		int min;
		int minIndex;
		Objects.requireNonNull(array);
		if ( array.length == 0 ) throw new NoSuchElementException​("PebbleSort.indexOfMin() require one element in the array");
		if ( index1 == index2 ) return array[index1];
		min = array[index1];
		minIndex = index1;
		for ( int i = index1 ; i < index2 ; i++ )
		{
			if ( min > array[i] )
			{
				min = array[i];
				minIndex = i;
			}
		}

		return minIndex;
	}

	public static void sort(int[] array)
	{
		for ( int i = 0 ; i < array.length ; i++ )
		{
			PebbleSort.swap(array,i,indexOfMin(array));
		}
	}

	public static void main(String[] args) {
		int[] test = new int[10];
		for ( int i = 9 ; i >= 0 ; i-- )
		{
			test[i] = i;
		}

		sort(test);
		for ( int i = 0 ; i < test.length ; i++ )
		{
			System.out.println(""+test[i]);
		}
	}
}